"""ProjectionPlot: a seeded 2D t-SNE projection of a table's embeddings,
colored by a label. Illustrative only - always report it next to a
quantitative analysis (ClusterAgreement, LabelCompactness)."""

import csv
import math
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.manifold import TSNE

from src.evaluation.analyses.card_sample import CardSample
from src.evaluation.analyses.embedding_analysis import (
    AnalysisResult,
    require_fresh_output_dir,
    write_analysis_result,
)
from src.evaluation.analyses.labeled_sample import LabeledSample, draw_labeled_sample
from src.evaluation.analyses.projection_renderer import ProjectionRenderer
from src.evaluation.embedding.embedding_table import EmbeddingTable
from src.evaluation.labels.card_labels import CardLabels

_OVERVIEW_FILE = "projection.png"
_PANELS_FILE = "projection_by_label.png"
_COORDINATES_FILE = "projection.csv"

# scikit-learn's TSNE refuses fewer iterations
_MIN_TSNE_ITERATIONS = 250


class ProjectionPlot:
    """t-SNE (scikit-learn) of the sampled cards' unit-length vectors,
    seeded and with fixed hyperparameters so plots are comparable across
    tables.

    Writes:
        projection.png: every card; the highlighted labels (see below)
            colored, the rest one neutral "Other" drawn beneath.
        projection_by_label.png: small multiples, one panel per label (the
            max_panels largest), that label highlighted over all cards.
        projection.csv: nocab_uuid, source_game, label, x, y per card - the
            table view, and a way to re-plot without re-running t-SNE.
        scalars.json: "kl_divergence" (t-SNE's final fit; left out if not
            finite, so the figures and scalars stay consistent), "n_cards",
            "n_labels".
    name = "projection".

    Which labels are colored in the overview, set by `highlighted`: a count
    colors that many most-sampled labels (ties by name); a tuple names the
    labels, in color-slot order. Under PerLabelCap(n) every label with >= n
    cards ties at n, so a count then picks alphabetically - name the labels
    to choose (and to keep the same choice across every encoder being
    compared). Panels always follow the count ranking.

    Inputs (constructor): labels, sample, seed (sampling and t-SNE),
        perplexity (> 0), max_iter (>= 250, scikit-learn's minimum),
        highlighted (None: as many as the style has series colors; a count;
        or a tuple of distinct labels; either way 1 to
        renderer.style.max_highlighted: a scatter keeps only that many
        colors distinguishable for color-blind readers), max_panels (>= 1),
        renderer (Strategy: how the figures are drawn; the reference
        palette by default).
    """

    def __init__(
        self,
        labels: CardLabels,
        sample: CardSample,
        seed: int,
        *,
        perplexity: float = 30.0,
        max_iter: int = 1000,
        highlighted: int | tuple[str, ...] | None = None,
        max_panels: int = 12,
        renderer: ProjectionRenderer = ProjectionRenderer(),
    ) -> None:
        """Side effects: none. Exceptions: ValueError on any out-of-range
        hyperparameter, or an invalid highlighted (count or tuple; see the
        class docstring)."""
        _require_hyperparameters(perplexity, max_iter, max_panels)
        max_highlighted = renderer.style.max_highlighted
        chosen = max_highlighted if highlighted is None else highlighted
        # A count is checked here; a label tuple by the renderer's own rule
        if isinstance(chosen, int):
            _require_highlighted_count(chosen, max_highlighted)
        else:
            renderer.require_highlightable(chosen)
        self.name = "projection"
        self._labels = labels
        self._sample = sample
        self._seed = seed
        self._perplexity = perplexity
        self._max_iter = max_iter
        self._highlighted = chosen
        self._max_panels = max_panels
        self._renderer = renderer

    def run(self, table: EmbeddingTable, output_dir: Path) -> AnalysisResult:
        """See EmbeddingAnalysis.run. Also raises ValueError (nothing
        written) if the sample has no more cards than perplexity
        (scikit-learn's t-SNE requires perplexity < n_cards).

        Example:
            >>> ProjectionPlot(GameLabels(), PerLabelCap(500), seed=0).run(
            ...     table, out / "projection").files
            (.../projection.png, .../projection_by_label.png,
             .../projection.csv, .../scalars.json)
        """
        require_fresh_output_dir(output_dir)
        sample = draw_labeled_sample(table, self._labels, self._sample, self._seed)
        _require_enough_cards(len(sample.rows), self._perplexity)

        # Project, then rank labels once for both figures
        coordinates, kl_divergence = self._project(sample.unit_vectors())
        ranked = _labels_by_count(sample.labels)
        highlighted = _highlighted_labels(self._highlighted, ranked)

        # Figures and the coordinate table, then scalars.json last
        output_dir.mkdir(parents=True, exist_ok=True)
        files = (
            self._renderer.draw_overview(
                coordinates,
                sample.labels,
                highlighted,
                output_dir / _OVERVIEW_FILE,
            ),
            self._renderer.draw_label_panels(
                coordinates,
                sample.labels,
                ranked[: self._max_panels],
                output_dir / _PANELS_FILE,
            ),
            _write_coordinates(sample, coordinates, output_dir / _COORDINATES_FILE),
        )
        scalars = sample.size_scalars()
        # A score that cannot be computed is left out, never written as NaN
        if math.isfinite(kl_divergence):
            scalars["kl_divergence"] = kl_divergence
        return write_analysis_result(output_dir, scalars, extra_files=files)

    def _project(self, vectors: np.ndarray) -> tuple[np.ndarray, float]:
        """Inputs: unit vectors (n, d), n > perplexity. Output: (n, 2)
        coordinates and the fit's KL divergence (TSNE with init="pca",
        random_state=seed, this plot's perplexity and max_iter). Side
        effects: none. Exceptions: none expected."""
        tsne = TSNE(
            n_components=2,
            perplexity=self._perplexity,
            max_iter=self._max_iter,
            init="pca",
            random_state=self._seed,
        )
        coordinates = tsne.fit_transform(vectors)
        return coordinates, float(tsne.kl_divergence_)


def _require_hyperparameters(perplexity: float, max_iter: int, max_panels: int) -> None:
    """Inputs: the constructor's numeric hyperparameters. Output: None.
    Side effects: none. Exceptions: ValueError naming the first
    out-of-range one (perplexity <= 0, max_iter < 250, max_panels < 1)."""
    if not perplexity > 0:  # also rejects NaN
        raise ValueError(f"perplexity must be > 0, got {perplexity}")
    if max_iter < _MIN_TSNE_ITERATIONS:
        raise ValueError(f"max_iter must be >= {_MIN_TSNE_ITERATIONS}, got {max_iter}")
    if max_panels < 1:
        raise ValueError(f"max_panels must be >= 1, got {max_panels}")


def _require_highlighted_count(count: int, max_highlighted: int) -> None:
    """Inputs: a highlighted count, the renderer's limit. Output: None.
    Side effects: none. Exceptions: ValueError if count is outside 1 to
    max_highlighted, or if count is a bool (Python treats True as 1)."""
    if isinstance(count, bool) or not 1 <= count <= max_highlighted:
        raise ValueError(
            f"highlighted must be a count from 1 to {max_highlighted}, got {count!r}"
        )


def _highlighted_labels(
    highlighted: int | tuple[str, ...], ranked: list[str]
) -> tuple[str, ...]:
    """Inputs: the constructor's highlighted, the count-ranked labels.
    Output: the labels to color, in slot order (a tuple as given; a count
    as that many from the front of ranked). Side effects: none.
    Exceptions: none."""
    if isinstance(highlighted, int):
        return tuple(ranked[:highlighted])
    return highlighted


def _require_enough_cards(card_count: int, perplexity: float) -> None:
    """Inputs: sampled card count, perplexity. Output: None. Side effects:
    none. Exceptions: ValueError naming both unless card_count > perplexity."""
    if card_count <= perplexity:
        raise ValueError(
            f"t-SNE needs more cards than perplexity: {card_count} sampled, "
            f"perplexity {perplexity}"
        )


def _labels_by_count(labels: tuple[str, ...]) -> list[str]:
    """Inputs: one label per card. Output: the distinct labels, most cards
    first, ties by name. Side effects: none. Exceptions: none."""
    counts = Counter(labels)
    return sorted(counts, key=lambda label: (-counts[label], label))


def _write_coordinates(
    sample: LabeledSample, coordinates: np.ndarray, path: Path
) -> Path:
    """Inputs: the sample, its (n, 2) coordinates, the CSV path. Output:
    path. Side effects: writes the CSV (header nocab_uuid, source_game,
    label, x, y; one row per card, sample order; nocab_uuid as its
    canonical string, source_game as its GameId value). Exceptions:
    OSError."""
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["nocab_uuid", "source_game", "label", "x", "y"])
        for row, label, (x, y) in zip(sample.rows, sample.labels, coordinates):
            writer.writerow(
                [str(row.nocab_uuid), row.source_game.value, label, float(x), float(y)]
            )
    return path
