"""read_csv_header - a 17lands CSV's column names, read from its first
line only (a chunk parser is built from it before any batch is read)."""

import csv
from pathlib import Path


def read_csv_header(csv_path: Path) -> tuple[str, ...]:
    """A CSV's column names, from its first line.

    Inputs: csv_path. Output: the column names, in file order.
    Side effects: reads csv_path's first line; the file is closed after.
    Exceptions: OSError reading csv_path; ValueError if it is empty.

    Example:
        >>> read_csv_header(Path("data/raw/17lands/game_data/KTK.TradSealed.csv"))[:2]
        ('expansion', 'event_type')
    """
    # utf-8-sig drops a byte-order mark, as pyarrow's reader does
    with csv_path.open(newline="", encoding="utf-8-sig") as raw_csv:
        header = next(csv.reader(raw_csv), None)
    if header is None:
        raise ValueError(f"{csv_path} is empty")
    return tuple(header)
