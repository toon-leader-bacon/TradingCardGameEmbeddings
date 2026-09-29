"""EmbeddingAnalysis: one intrinsic measurement over a stored EmbeddingTable,
and the result every analysis returns and writes.

Every analysis shares one call - run(table, output_dir). Anything else it
needs (labels, a sampling rule, a seed, hyperparameters) is constructor
input, so analyses with and without labels share the Protocol. The two
output helpers here keep the on-disk contract identical across analyses.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Protocol

from src.evaluation.embedding.embedding_table import EmbeddingTable

_SCALARS_FILE = "scalars.json"


@dataclass(frozen=True)
class AnalysisResult:
    """What one run produced.

    files: every file written, scalars.json included.
    scalars: the analysis's numbers (e.g. {"ari": 0.41, "nmi": 0.52}),
        stored as a read-only copy; also written as scalars.json.
    """

    files: tuple[Path, ...]
    scalars: Mapping[str, float]

    def __post_init__(self) -> None:
        # Freeze scalars as a read-only copy (the dataclass is frozen)
        object.__setattr__(self, "scalars", MappingProxyType(dict(self.scalars)))


class EmbeddingAnalysis(Protocol):
    """One intrinsic measurement over a table.

    name: the analysis's directory name under
        <output_dir>/intrinsic/<encoder_label>/ (e.g. "cluster_agreement").
    """

    name: str

    def run(self, table: EmbeddingTable, output_dir: Path) -> AnalysisResult:
        """Measure table, writing into output_dir (this analysis's own
        directory; run creates it).

        Inputs: table (an open EmbeddingTable), output_dir (Path).
        Output: AnalysisResult.
        Side effects: creates output_dir and writes its files.
        Exceptions: FileExistsError if output_dir exists and is non-empty;
            ValueError if, after sampling and dropping unlabeled cards,
            fewer than two cards remain, or (label-using analyses) fewer
            than two distinct labels, or an analysis-specific requirement
            fails - nothing is written in any of these cases.
        """
        ...


def require_fresh_output_dir(output_dir: Path) -> None:
    """Refuse to overwrite an earlier run's results.

    Inputs: output_dir (Path). Output: None. Side effects: none (the
    directory is created later, by write_analysis_result).
    Exceptions: FileExistsError if output_dir exists and is non-empty (an
        existing empty directory is fine).

    Example:
        >>> require_fresh_output_dir(Path("out/intrinsic/a/cluster_agreement"))
    """
    # A file at the path, or a directory holding anything, is an earlier run
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(f"{output_dir} already holds results")


def write_analysis_result(
    output_dir: Path,
    scalars: Mapping[str, float],
    extra_files: tuple[Path, ...] = (),
) -> AnalysisResult:
    """Write scalars.json into output_dir and assemble the result.

    Inputs: output_dir (Path); scalars; extra_files already written into
        output_dir by the analysis (e.g. a plot).
    Output: AnalysisResult with files = extra_files + (scalars.json,).
    Side effects: creates output_dir (and parents) if missing; writes
        scalars.json (keys sorted, indented).
    Exceptions: ValueError (nothing written) if any scalar is NaN or
        infinite - an analysis leaves a score it cannot compute out rather
        than writing non-standard JSON; OSError on a write failure.

    Example:
        >>> write_analysis_result(out, {"ari": 0.41}).files
        (PosixPath('out/scalars.json'),)
    """
    # allow_nan=False raises ValueError on NaN/inf before anything is written
    text = json.dumps(dict(scalars), indent=2, sort_keys=True, allow_nan=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    scalars_path = output_dir / _SCALARS_FILE
    scalars_path.write_text(text)
    return AnalysisResult(files=extra_files + (scalars_path,), scalars=scalars)
