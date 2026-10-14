"""Summarize normalized TEST loss per round for every finished cell.

Reads each cell's rounds.csv and run_config.yaml under
data/runs/2026-10-11_model_by_diet_r2/<cell>/ and writes, into this
experiment's results/:

    test_loss_by_round.csv   one row per (cell, round, kind): median,
                             quartiles, counts below 1x and above 5x, max
    test_loss_by_dojo.csv    one row per (cell, dojo): kind and the
                             normalized TEST loss of every round
    test_loss_by_round.png   median per round with the interquartile band,
                             one panel per kind, one line per cell

Kinds: "contrastive" (1.00 = chance), "metric" (trained metric dojos;
1.0 = mean predictor) and "held out" (the config's held_out_dojos).

Usage (from the project root, ROCm venv):
    python experiments/2026-10-11_model_by_diet_r2/analysis/test_loss.py
"""

import csv
import statistics
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import yaml  # noqa: E402

RUNS = Path("data/runs/2026-10-11_model_by_diet_r2")
RESULTS = Path("experiments/2026-10-11_model_by_diet_r2/results")
CELLS = (
    "single_contrastive",
    "single_metric",
    "single_mixed",
    "multi_contrastive",
    "multi_metric",
    "multi_mixed",
)
# Fixed colour per cell (validated default categorical palette, slots 1-6),
# so a cell keeps its colour as more cells finish.
CELL_COLOURS = {
    "single_contrastive": "#2a78d6",
    "single_metric": "#eb6834",
    "single_mixed": "#1baf7a",
    "multi_contrastive": "#eda100",
    "multi_metric": "#e87ba4",
    "multi_mixed": "#008300",
}
KINDS = ("contrastive", "metric", "held out")
OVERFLOW_RATIO = 5.0


@dataclass(frozen=True)
class DojoRound:
    """One dojo's TEST result for one round of one cell."""

    cell: str
    round_index: int
    step: int
    elapsed_minutes: float
    dojo: str
    kind: str
    normalized_loss: float


def kind_of(dojo: str, held_out: frozenset[str]) -> str:
    """The dojo's kind: contrastive, held out, or (trained) metric."""
    if dojo in held_out:
        return "held out"
    return "contrastive" if dojo.startswith("contrastive") else "metric"


def read_cell(cell: str) -> list[DojoRound]:
    """Every (round, dojo) row of one finished cell; empty if not run yet."""
    rounds_path = RUNS / cell / "rounds.csv"
    if not rounds_path.exists():
        return []
    config = yaml.safe_load((RUNS / cell / "run_config.yaml").read_text())
    held_out = frozenset(config.get("held_out_dojos", ()))
    with rounds_path.open(newline="") as handle:
        return [
            DojoRound(
                cell=cell,
                round_index=int(row["round_index"]),
                step=int(row["step"]),
                elapsed_minutes=float(row["elapsed_seconds"]) / 60,
                dojo=row["dojo"],
                kind=kind_of(row["dojo"], held_out),
                normalized_loss=float(row["normalized_test_loss"]),
            )
            for row in csv.DictReader(handle)
        ]


def summary_rows(results: list[DojoRound]) -> list[dict[str, object]]:
    """One summary row per (cell, round, kind) present in results."""
    keys = sorted({(r.cell, r.round_index, r.kind) for r in results})
    rows: list[dict[str, object]] = []
    for cell, round_index, kind in keys:
        group = [
            r
            for r in results
            if (r.cell, r.round_index, r.kind) == (cell, round_index, kind)
        ]
        losses = sorted(r.normalized_loss for r in group)
        q1, median, q3 = statistics.quantiles(losses, n=4, method="inclusive")
        rows.append(
            {
                "cell": cell,
                "round": round_index,
                "step": group[0].step,
                "elapsed_min": round(group[0].elapsed_minutes, 1),
                "kind": kind,
                "n_dojos": len(losses),
                "median": round(median, 4),
                "q25": round(q1, 4),
                "q75": round(q3, 4),
                "n_below_1": sum(x < 1 for x in losses),
                "n_above_5": sum(x > OVERFLOW_RATIO for x in losses),
                "max": round(losses[-1], 4),
            }
        )
    return rows


def by_dojo_rows(results: list[DojoRound]) -> list[dict[str, object]]:
    """One row per (cell, dojo) with each round's normalized loss."""
    rounds = sorted({r.round_index for r in results})
    table: dict[tuple[str, str], dict[str, object]] = {}
    for r in results:
        row = table.setdefault(
            (r.cell, r.dojo), {"cell": r.cell, "dojo": r.dojo, "kind": r.kind}
        )
        row[f"round{r.round_index}"] = round(r.normalized_loss, 4)
    for row in table.values():
        for number in rounds:
            row.setdefault(f"round{number}", "")
    return [table[key] for key in sorted(table)]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write rows (all sharing the first row's keys) to path."""
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict[str, object]], path: Path) -> None:
    """Median per round with the interquartile band; one panel per kind."""
    figure, axes = plt.subplots(1, len(KINDS), figsize=(12, 3.8), sharex=True)
    cells = [c for c in CELLS if any(row["cell"] == c for row in rows)]
    for axis, kind in zip(axes, KINDS):
        for cell in cells:
            series = [r for r in rows if r["cell"] == cell and r["kind"] == kind]
            if not series:
                continue
            steps = [int(str(r["step"])) for r in series]
            colour = CELL_COLOURS[cell]
            axis.fill_between(
                steps,
                [float(str(r["q25"])) for r in series],
                [float(str(r["q75"])) for r in series],
                color=colour,
                alpha=0.15,
                linewidth=0,
            )
            axis.plot(
                steps,
                [float(str(r["median"])) for r in series],
                color=colour,
                linewidth=2,
                marker="o",
                markersize=5,
                label=f"{cell} (n={series[0]['n_dojos']})",
            )
        axis.axhline(1.0, color="#52514e", linewidth=1, linestyle="--")
        reference = "chance" if kind == "contrastive" else "mean predictor"
        axis.set_title(f"{kind} (1.0 = {reference})", fontsize=10)
        axis.set_xlabel("training step")
        axis.grid(axis="y", color="#e5e4e0", linewidth=0.8)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        axis.legend(fontsize=8, frameon=False)
    axes[0].set_ylabel("normalized TEST loss (median, IQR band)")
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


def main() -> None:
    """Read every finished cell, then write the tables and the figure."""
    results = [row for cell in CELLS for row in read_cell(cell)]
    if not results:
        raise SystemExit(f"no finished cells under {RUNS}")
    RESULTS.mkdir(parents=True, exist_ok=True)
    summary = summary_rows(results)
    write_csv(RESULTS / "test_loss_by_round.csv", summary)
    write_csv(RESULTS / "test_loss_by_dojo.csv", by_dojo_rows(results))
    plot(summary, RESULTS / "test_loss_by_round.png")
    for row in summary:
        print(
            f"{row['cell']:20s} r{row['round']} {row['kind']:12s} "
            f"n={row['n_dojos']:3} median={row['median']:.3f} "
            f"IQR {row['q25']:.3f}-{row['q75']:.3f} <1:{row['n_below_1']:3} "
            f">5x:{row['n_above_5']} max={row['max']:.2f}"
        )


if __name__ == "__main__":
    main()
