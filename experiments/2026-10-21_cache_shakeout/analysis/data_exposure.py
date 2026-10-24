"""How fast a run at a measured step rate passes over each dojo's TRAIN data.

For each trained dojo: its diet share and TRAIN count (the run's
diet_shares.csv; decks for contrastive dojos, rows for metric dojos), the
examples per batch it actually built (profile/<run>/builds.csv), and the
run's measured training steps per hour (steps.csv) give

    examples per hour = share * steps per hour * examples per batch
    hours per pass    = TRAIN count / examples per hour

A dojo with few hours per pass repeats its data many times in a long run:
more data there is worth the most. Writes results/data_exposure_<run>.csv
and results/data_exposure_by_family_<run>.csv.

Usage (project root): python experiments/2026-10-21_cache_shakeout/analysis/data_exposure.py <run> [...]
"""

import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

RUNS = Path("data/runs/2026-10-21_cache_shakeout")
RESULTS = Path("experiments/2026-10-21_cache_shakeout/results")


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def family(dojo: str) -> str:
    source = dojo.split(".", 1)[0]
    return source if source.startswith("contrastive") else f"metric:{source}"


def report(run: str) -> None:
    shares = rows(RUNS / run / "diet_shares.csv")
    steps = rows(RUNS / "profile" / run / "steps.csv")
    builds = [b for b in rows(RUNS / "profile" / run / "builds.csv") if b["build_s"] != ""]
    seconds = sum(float(s["total_s"]) for s in steps)
    steps_per_hour = len(steps) / seconds * 3600
    examples: dict[str, list[float]] = defaultdict(list)
    for b in builds:
        if b.get("examples"):
            examples[b["dojo"]].append(float(b["examples"]))
    table = []
    for row in shares:
        dojo, count, share = row["dojo"], float(row["train_count"]), float(row["share"])
        per_batch = statistics.mean(examples[dojo]) if examples[dojo] else float("nan")
        per_hour = share * steps_per_hour * per_batch
        hours = count / per_hour if per_hour == per_hour and per_hour > 0 else float("nan")
        table.append({
            "dojo": dojo, "family": family(dojo), "train_count": int(count), "share": share,
            "examples_per_batch": per_batch, "examples_per_hour": per_hour,
            "hours_per_pass": hours, "passes_per_24h": 24 / hours if hours == hours else float("nan"),
        })
    table.sort(key=lambda r: (r["hours_per_pass"] != r["hours_per_pass"], r["hours_per_pass"]))
    RESULTS.mkdir(parents=True, exist_ok=True)
    fields = list(table[0])
    with (RESULTS / f"data_exposure_{run}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for r in table:
            writer.writerow({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in r.items()})

    # Per family: median and min hours per pass, and the TRAIN total
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for r in table:
        groups[str(r["family"])].append(r)
    with (RESULTS / f"data_exposure_by_family_{run}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["family", "dojos", "train_total", "share_total", "hours_per_pass_min",
                         "hours_per_pass_median", "steps_per_hour"])
        for name, group in sorted(groups.items()):
            hours = [float(r["hours_per_pass"]) for r in group if r["hours_per_pass"] == r["hours_per_pass"]]
            writer.writerow([name, len(group), sum(int(r["train_count"]) for r in group),
                             f"{sum(float(r['share']) for r in group):.4f}",
                             f"{min(hours):.4g}" if hours else "", f"{statistics.median(hours):.4g}" if hours else "",
                             f"{steps_per_hour:.0f}"])
    print(f"{run}: {steps_per_hour:.0f} steps/h; {len(table)} dojos")


if __name__ == "__main__":
    for name in sys.argv[1:]:
        report(name)
