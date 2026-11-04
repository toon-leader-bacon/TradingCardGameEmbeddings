"""Two figures from results/run_summary.csv, one bar per run:

- results/wall_time_by_phase.png: where a run's wall time went (startup
  build_dojos, the rest of startup, training steps, TEST passes), minutes.
- results/step_time_by_part.png: where a training step's time went (live
  encoding, the rest of forward + loss, backward + optimizer, batch wait), ms.

Usage (project root): python experiments/2026-10-21_cache_shakeout/analysis/plot_timings.py [runs]
"""

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = Path("experiments/2026-10-21_cache_shakeout/results")
DEFAULT_RUNS = ["cached", "cached_nocopy", "cached_nomods", "uncached", "cached_seed1"]
# Reference categorical palette (dataviz skill), slots 1-4 in fixed order, light mode
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"


def summaries(runs: list[str]) -> list[dict[str, str]]:
    with (RESULTS / "run_summary.csv").open(encoding="utf-8", newline="") as handle:
        by_run = {row["run"]: row for row in csv.DictReader(handle)}
    return [by_run[r] for r in runs if r in by_run]


def number(row: dict[str, str], key: str) -> float:
    return float(row[key]) if row.get(key) else 0.0


def stacked(rows: list[dict[str, str]], parts: dict[str, list[float]], unit: str, title: str,
            path: Path) -> None:
    labels = [r["run"] for r in rows]
    figure, axes = plt.subplots(figsize=(9, 0.6 * len(rows) + 1.6), facecolor=SURFACE)
    axes.set_facecolor(SURFACE)
    left = [0.0] * len(rows)
    for color, (name, values) in zip(SERIES, parts.items()):
        axes.barh(labels, values, left=left, color=color, height=0.55, label=name,
                  edgecolor=SURFACE, linewidth=2)
        left = [a + b for a, b in zip(left, values)]
    for y, total in enumerate(left):
        axes.text(total, y, f"  {total:.0f} {unit}", va="center", color=INK, fontsize=9)
    axes.invert_yaxis()
    axes.set_xlim(0, max(left) * 1.15)
    axes.set_xlabel(unit, color=MUTED)
    axes.set_title(title, color=INK, loc="left", fontsize=11)
    axes.tick_params(colors=MUTED)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    axes.grid(axis="x", color="#e4e3df", linewidth=0.8)
    axes.set_axisbelow(True)
    axes.legend(loc="lower center", bbox_to_anchor=(0.5, 1.08), ncol=len(parts), frameon=False,
                fontsize=9, labelcolor=INK)
    figure.tight_layout()
    figure.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(figure)
    print(path)


def main() -> None:
    rows = summaries(sys.argv[1:] or DEFAULT_RUNS)
    minutes = lambda key: [number(r, key) / 60 for r in rows]  # noqa: E731
    other_startup = [
        (number(r, "startup_preflight_s") + number(r, "startup_build_model_s")
         + number(r, "startup_warm_cache_s")) / 60 for r in rows]
    test = [(number(r, "test_round0_s") + number(r, "test_round1_s")) / 60 for r in rows]
    stacked(rows, {
        "build_dojos": minutes("startup_build_dojos_s"),
        "other startup": other_startup,
        "training steps": minutes("steps_s_total"),
        "TEST passes (2)": test,
    }, "min", "Wall time per run (2,000 steps)", RESULTS / "wall_time_by_phase.png")

    step_ms = [number(r, "step_mean_s") * 1000 for r in rows]
    live = [s * number(r, "live_share_of_step") for s, r in zip(step_ms, rows)]
    forward = [s * number(r, "forward_loss_share") for s, r in zip(step_ms, rows)]
    stacked(rows, {
        "live encoding": live,
        "rest of forward + loss": [f - l for f, l in zip(forward, live)],
        "backward + optimizer": [s * number(r, "backward_opt_share") for s, r in zip(step_ms, rows)],
        "batch wait": [s * number(r, "wait_share") for s, r in zip(step_ms, rows)],
    }, "ms", "Mean training step, by part", RESULTS / "step_time_by_part.png")


if __name__ == "__main__":
    main()
