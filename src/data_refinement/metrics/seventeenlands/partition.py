"""SeventeenLandsPartition - where one 17lands metric's output for one
source CSV lives, and the set and format it covers.

The partition path is the single source of truth for a file's set and
format; the file itself never repeats them. Every 17lands metric output
lives at

    <metrics_root>/seventeenlands/<family>/<metric_stem>/<SET>/<Format>.parquet

The run driver (scripts/run_metrics.py) builds that path with path();
the slice builder (slice_file.py) lists a metric's files with
find_partitions() and reads each one's set and format back with
from_path(). Neither ever parses the layout itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.data_retrieval.seventeenlands.refs import DataType, Expansion, format_code

METRICS_ROOT = Path("data/metrics")
SOURCE_DIRECTORY = "seventeenlands"
PARTITION_SUFFIX = ".parquet"
# Built slice files live in <family>/slices/ (slice_file.py), beside the
# metric directories, so no metric may use this as its stem.
SLICES_DIRECTORY = "slices"


@dataclass(frozen=True)
class SeventeenLandsPartition:
    """One (family, metric, set, format) output file.

    family: which raw export the metric reads.
    metric_stem: the metric's OUTPUT_STEM, e.g. "drawn_win_rate".
    expansion: the source CSV's set.
    format: the source CSV's event format.
    """

    family: DataType
    metric_stem: str
    expansion: Expansion
    format: format_code

    def __post_init__(self) -> None:
        """Reject a stem that can't be a metric directory.

        Inputs: none. Output: none. Side effects: none.
        Exceptions: ValueError if metric_stem is empty, contains a path
            separator or ".", or is SLICES_DIRECTORY.
        """
        stem = self.metric_stem
        if not stem or stem == SLICES_DIRECTORY or any(c in stem for c in "/\\."):
            raise ValueError(f"{stem!r} is not a valid metric stem")

    def path(self, metrics_root: Path = METRICS_ROOT) -> Path:
        """This partition's file path.

        Inputs: metrics_root (data/metrics/, or a scratch root for a
            parity run).
        Output: <metrics_root>/seventeenlands/<family>/<metric_stem>/
            <SET>/<Format>.parquet.
        Side effects: none. Exceptions: none.

        Example:
            >>> SeventeenLandsPartition(
            ...     DataType.GAME, "drawn_win_rate", Expansion.KTK, format_code.TradDraft
            ... ).path()
            PosixPath('data/metrics/seventeenlands/game_data/drawn_win_rate/KTK/TradDraft.parquet')
        """
        return (
            metrics_root
            / SOURCE_DIRECTORY
            / self.family.value
            / self.metric_stem
            / self.expansion.value
            / f"{self.format.value}{PARTITION_SUFFIX}"
        )

    @classmethod
    def from_path(cls, path: Path) -> SeventeenLandsPartition:
        """The inverse of path(): read a partition back from its file
        path, under any metrics root.

        Inputs: path, a file path ending in
            seventeenlands/<family>/<metric_stem>/<SET>/<Format>.parquet.
        Output: the partition it names.
        Side effects: none (the path is parsed, never opened).
        Exceptions: ValueError if path is not of that shape, or names an
            unknown family, set or format.

        Example:
            >>> root = Path("data/metrics/seventeenlands/game_data")
            >>> SeventeenLandsPartition.from_path(
            ...     root / "drawn_win_rate" / "KTK" / "TradDraft.parquet"
            ... ).expansion
            <Expansion.KTK: 'KTK'>
        """
        # Validate the shape: .parquet, under a seventeenlands/ directory
        if path.suffix != PARTITION_SUFFIX:
            raise ValueError(f"{path} is not a {PARTITION_SUFFIX} partition file")
        family_name, metric_stem, expansion_code = _trailing_directories(path, 3)

        # Parse each segment into its enum (ValueError on an unknown one)
        return cls(
            family=DataType(family_name),
            metric_stem=metric_stem,
            expansion=Expansion(expansion_code),
            format=format_code(path.stem),
        )


def find_partitions(
    family: DataType, metric_stem: str, metrics_root: Path = METRICS_ROOT
) -> list[SeventeenLandsPartition]:
    """Every partition file on disk for one metric.

    Inputs: family, metric_stem, metrics_root.
    Output: the partitions, sorted by (set, format) value; [] if the
        metric has no output yet.
    Side effects: lists directories under metrics_root (no file is
        opened).
    Exceptions: ValueError if a file under the metric's directory is
        not a valid partition path (a stray file is a layout bug, never
        silently skipped).

    Example:
        >>> find_partitions(DataType.GAME, "drawn_win_rate")
        [SeventeenLandsPartition(family=<DataType.GAME: 'game_data'>, ...), ...]
    """
    result: list[SeventeenLandsPartition] = []
    metric_directory = metrics_root / SOURCE_DIRECTORY / family.value / metric_stem
    if not metric_directory.is_dir():
        return result

    # Each <SET>/<Format>.parquet under the metric's directory
    for path in metric_directory.glob(f"*/*{PARTITION_SUFFIX}"):
        result.append(SeventeenLandsPartition.from_path(path))

    result.sort(
        key=lambda partition: (partition.expansion.value, partition.format.value)
    )
    return result


def _trailing_directories(path: Path, count: int) -> tuple[str, ...]:
    """The names of path's last `count` parent directories, outermost
    first, checked to sit directly under a seventeenlands/ directory.

    Inputs: path, count (>= 1).
    Output: count directory names.
    Side effects: none.
    Exceptions: ValueError if path has too few parents or the directory
        above them is not seventeenlands/.
    """
    parents = path.parent.parts
    if len(parents) < count + 1 or parents[-count - 1] != SOURCE_DIRECTORY:
        raise ValueError(
            f"{path} is not under {SOURCE_DIRECTORY}/ with {count} directories"
        )
    return tuple(parents[-count:])
