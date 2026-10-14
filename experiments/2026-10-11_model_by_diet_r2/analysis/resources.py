"""Summarize each finished cell's timing and resource use.

Reads, per cell, under data/runs/2026-10-11_model_by_diet_r2/:
    queue.csv                       start and end of the run
    <cell>/run.log                  when training began (the trainable-
                                    parameters line) and when each round's
                                    TEST ended
    logs/<cell>.resources.csv       60 s samples: free RAM, Python RAM/VRAM
    logs/<cell>.utilization.csv     30 s samples: GPU engine busy %, Python CPU %

and writes results/resources_by_cell.csv: one row per cell with the run's
duration, startup, minutes per round, peak memory, and medians and p10/p90
of GPU compute busy % and Python CPU % during training (after startup).
A cell whose sample files are missing or were lost gets blank columns.

Usage (from the project root, ROCm venv):
    python experiments/2026-10-11_model_by_diet_r2/analysis/resources.py
"""

import csv
import re
import statistics
from datetime import datetime
from pathlib import Path

RUNS = Path("data/runs/2026-10-11_model_by_diet_r2")
RESULTS = Path("experiments/2026-10-11_model_by_diet_r2/results")
# multi_contrastive's sample files were overwritten by a launcher bug
# (bench note 2026-10-12_1500_incident_launcher_rerun.md); its samples are
# reported as missing rather than read.
LOST_SAMPLES = {"multi_contrastive"}
LOG_TIME = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})")
ROUND_LINE = re.compile(r"\[train\] round (\d+) \(step \d+\) test loss")


def log_time(line: str) -> datetime:
    """The timestamp at the start of a run.log line."""
    match = LOG_TIME.match(line)
    if match is None:
        raise ValueError(f"no timestamp: {line[:60]!r}")
    return datetime.fromisoformat(match.group(1))


def training_marks(run_log: Path) -> tuple[datetime, list[datetime]]:
    """When training began, and when each round's TEST ended."""
    start: datetime | None = None
    round_ends: list[datetime] = []
    for line in run_log.read_text(encoding="utf-8", errors="replace").splitlines():
        if start is None and "trainable parameters" in line:
            start = log_time(line)
        elif ROUND_LINE.search(line):
            round_ends.append(log_time(line))
    if start is None:
        raise ValueError(f"no training start in {run_log}")
    return start, round_ends


def sample_rows(path: Path) -> list[dict[str, str]]:
    """The rows of one sampler CSV (written with a UTF-8 BOM)."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def spread(values: list[float]) -> tuple[float, float, float]:
    """(p10, median, p90) of values."""
    ordered = sorted(values)
    return (
        ordered[len(ordered) // 10],
        statistics.median(ordered),
        ordered[9 * len(ordered) // 10],
    )


def cell_row(queue_row: dict[str, str]) -> dict[str, object]:
    """One output row for a finished cell."""
    cell = queue_row["run"]
    started = datetime.fromisoformat(queue_row["started"])
    ended = datetime.fromisoformat(queue_row["ended"])
    train_start, round_ends = training_marks(RUNS / cell / "run.log")
    edges = [train_start, *round_ends]
    row: dict[str, object] = {
        "cell": cell,
        "duration_min": round((ended - started).total_seconds() / 60),
        "startup_min": round((train_start - started).total_seconds() / 60),
        "round_min": " ".join(
            str(round((b - a).total_seconds() / 60)) for a, b in zip(edges, edges[1:])
        ),
    }
    blank = dict.fromkeys(
        (
            "peak_py_ram_gb",
            "min_free_ram_gb",
            "peak_py_vram_gb",
            "peak_card_vram_gb",
            "gpu_compute_p10",
            "gpu_compute_median",
            "gpu_compute_p90",
            "py_cpu_p10",
            "py_cpu_median",
            "py_cpu_p90",
            "n_util_samples",
        ),
        "",
    )
    if cell in LOST_SAMPLES:
        return row | blank
    resources = sample_rows(RUNS / "logs" / f"{cell}.resources.csv")
    row |= blank | {
        "peak_py_ram_gb": max(float(r["python_ws_gb"]) for r in resources),
        "min_free_ram_gb": min(float(r["free_ram_gb"]) for r in resources),
        "peak_py_vram_gb": max(float(r["python_vram_gb"] or 0) for r in resources),
        "peak_card_vram_gb": max(float(r["total_vram_gb"] or 0) for r in resources),
    }
    # The utilization sampler started with multi_contrastive; earlier cells have none
    utilization_path = RUNS / "logs" / f"{cell}.utilization.csv"
    if not utilization_path.exists():
        return row
    utilization = [
        r
        for r in sample_rows(utilization_path)
        if datetime.fromisoformat(r["time"]) > train_start
    ]
    compute = spread([float(r["gpu_compute_pct"]) for r in utilization])
    cpu = spread([float(r["py_cpu_pct"]) for r in utilization])
    return row | {
        "gpu_compute_p10": round(compute[0], 1),
        "gpu_compute_median": round(compute[1], 1),
        "gpu_compute_p90": round(compute[2], 1),
        "py_cpu_p10": round(cpu[0], 1),
        "py_cpu_median": round(cpu[1], 1),
        "py_cpu_p90": round(cpu[2], 1),
        "n_util_samples": len(utilization),
    }


def main() -> None:
    """Write results/resources_by_cell.csv for every finished cell."""
    with (RUNS / "queue.csv").open(newline="") as handle:
        finished = [r for r in csv.DictReader(handle) if r["status"] == "done"]
    rows = [cell_row(r) for r in finished]
    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / "resources_by_cell.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
