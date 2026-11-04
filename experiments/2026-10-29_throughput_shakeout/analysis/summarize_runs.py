"""One row per run: timing, throughput, cache and resources -> results/runs.csv.

Usage (from project root):
    python experiments/2026-10-29_throughput_shakeout/analysis/summarize_runs.py

Reads, per run in queue.csv with status done:
- <run>/rounds.csv: elapsed seconds at the end of each round (TEST included);
- <run>/run_config.yaml: steps per round and max_batch_cost;
- <run>/samples.csv: torch VRAM and feature cache counters (last row);
- <run>/run.log: start time and the first training line (startup);
- logs/<run>.utilization.csv and .resources.csv: Windows counters, clipped
  to the training window.

TRAIN cards per round = steps_per_round * max_batch_cost: single-card
contrastive batches are filled to the budget with real cards (2 per deck).
"steady" uses every round after the first.
"""

import csv
import re
import statistics
from datetime import datetime, timedelta
from pathlib import Path

import yaml

_RUNS = Path("data/runs/2026-10-29_throughput_shakeout")
_RESULTS = Path("experiments/2026-10-29_throughput_shakeout/results")
_LOG_TIME = re.compile(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)")


def _round_ends(run: str) -> list[float]:
    """Elapsed seconds at each round's end, one per round."""
    ends: dict[int, float] = {}
    for row in csv.DictReader(open(_RUNS / run / "rounds.csv", encoding="utf-8")):
        ends[int(row["round_index"])] = float(row["elapsed_seconds"])
    return [ends[index] for index in sorted(ends)]


def _log_times(run: str) -> tuple[datetime, datetime, datetime]:
    """Process start, first training line (the trainer's parameter line)
    and last line of run.log."""
    start = first = last = None
    for line in open(_RUNS / run / "run.log", encoding="utf-8", errors="replace"):
        match = _LOG_TIME.match(line)
        if match is None:
            continue
        stamp = datetime.fromisoformat(match.group(1))
        start = start or stamp
        if first is None and "trainable parameters: encoder" in line:
            first = stamp
        last = stamp
    assert start and first and last, run
    return start, first, last


def _counter_medians(run: str, start: datetime, end: datetime) -> dict[str, float]:
    """Median python CPU and GPU compute, and min free RAM / max working
    set, over [start, end]."""
    def rows(suffix: str) -> list[dict[str, str]]:
        path = _RUNS / "logs" / f"{run}.{suffix}.csv"
        return [
            row
            for row in csv.DictReader(open(path, encoding="utf-8-sig"))
            if start <= datetime.fromisoformat(row["time"]) <= end
        ]

    utilization = rows("utilization")
    resources = rows("resources")

    def number(text: str) -> float:
        return float(text.replace(",", ""))

    return {
        "py_cpu_pct_median": statistics.median(number(r["py_cpu_pct"]) for r in utilization),
        "gpu_compute_pct_median": statistics.median(
            number(r["gpu_compute_pct"]) for r in utilization
        ),
        "free_ram_gb_min": min(number(r["free_ram_gb"]) for r in resources),
        "python_ws_gb_max": max(number(r["python_ws_gb"]) for r in resources),
        "python_vram_gb_max": max(number(r["python_vram_gb"]) for r in resources),
    }


def summarize(run: str) -> dict[str, object]:
    """One run's row."""
    config = yaml.safe_load(open(_RUNS / run / "run_config.yaml", encoding="utf-8"))
    phase = config["phases"][0]
    cards_per_round = phase["steps_per_round"] * config["hardware"]["max_batch_cost"]
    ends = _round_ends(run)
    durations = [ends[0]] + [b - a for a, b in zip(ends, ends[1:])]
    steady = durations[1:] or durations
    start, first, last = _log_times(run)
    samples = list(csv.DictReader(open(_RUNS / run / "samples.csv", encoding="utf-8")))
    final = samples[-1]
    # Counters over the training window, skipping the first 30 s of it
    counters = _counter_medians(run, first + timedelta(seconds=30), last)
    return {
        "run": run,
        "max_batch_cost": config["hardware"]["max_batch_cost"],
        "dojos": len(config["dojos"]),
        "rounds": len(ends),
        "startup_s": (first - start).total_seconds(),
        "round_s": " ".join(f"{d:.0f}" for d in durations),
        "first_round_cards_per_s": round(cards_per_round / durations[0]),
        "steady_cards_per_s": round(cards_per_round / statistics.mean(steady)),
        "steady_ms_per_step": round(1000 * statistics.mean(steady) / phase["steps_per_round"], 1),
        "vram_peak_gib": round(max(float(r["vram_peak_gib"]) for r in samples), 2),
        "vram_reserved_gib": round(max(float(r["vram_reserved_gib"]) for r in samples), 2),
        "cache_rows": int(final["vram_rows"]),
        "cache_encoded": int(final["cache_encoded"]),
        "cache_uncacheable": int(final["cache_uncacheable"]),
        "evictions": int(final["vram_evicted"]) + int(final["ram_evicted"]),
        **{key: round(value, 2) for key, value in counters.items()},
    }


def main() -> None:
    queue = csv.DictReader(open(_RUNS / "queue.csv", encoding="utf-8"))
    rows = [summarize(row["run"]) for row in queue if row["status"] == "done"]
    _RESULTS.mkdir(parents=True, exist_ok=True)
    with open(_RESULTS / "runs.csv", "w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
