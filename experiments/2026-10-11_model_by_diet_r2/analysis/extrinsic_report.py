"""Summary table and figure for the fresh-head runs on the held-out dojos.

Reads results/extrinsic_validation.csv (extrinsic_held_out.py) and writes:

    results/extrinsic_summary.csv   per encoder: median normalized
                                    VALIDATION loss over the 13 dojos, how
                                    many dojos beat the baseline (< 1.0),
                                    and the median paired difference
                                    against its model shape's control
    results/extrinsic_by_dojo.png   per dojo, every encoder's normalized
                                    VALIDATION loss (1.0 = mean predictor /
                                    prior; lower is better)

Run (from the project root):
    python experiments/2026-10-11_model_by_diet_r2/analysis/extrinsic_report.py
"""

import csv
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = Path("experiments/2026-10-11_model_by_diet_r2/results")
DIET_COLOURS = {"contrastive": "#2a78d6", "metric": "#eb6834", "mixed": "#1baf7a"}
CONTROL_COLOUR = "#8a8984"
INK = "#2b2a27"
GRID = "#e5e4e0"
ENCODERS = (
    "single_contrastive", "single_metric", "single_mixed", "control_single_s0",
    "multi_contrastive", "multi_metric", "multi_mixed", "control_multi_s0",
)


def colour_of(encoder: str) -> str:
    return CONTROL_COLOUR if encoder.startswith("control") else DIET_COLOURS[encoder.split("_")[1]]


def main() -> None:
    with (RESULTS / "extrinsic_validation.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    loss = {(r["encoder"], r["dojo"]): float(r["validation_normalized"]) for r in rows}
    dojos = sorted({r["dojo"] for r in rows})
    summary = []
    for encoder in ENCODERS:
        values = [loss[(encoder, d)] for d in dojos if (encoder, d) in loss]
        control = "control_single_s0" if encoder.startswith(("single", "control_single")) else "control_multi_s0"
        diffs = [loss[(encoder, d)] - loss[(control, d)] for d in dojos
                 if (encoder, d) in loss and (control, d) in loss]
        summary.append({
            "encoder": encoder,
            "n_dojos": len(values),
            "median_normalized": round(statistics.median(values), 4),
            "n_below_1": sum(v < 1 for v in values),
            "median_minus_control": round(statistics.median(diffs), 4),
            "n_better_than_control_by_0.02": sum(d < -0.02 for d in diffs),
            "n_worse_than_control_by_0.02": sum(d > 0.02 for d in diffs),
        })
    with (RESULTS / "extrinsic_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    for row in summary:
        print(row)

    order = sorted(dojos, key=lambda d: loss.get(("control_single_s0", d), 1.0))
    figure, axis = plt.subplots(figsize=(10, 6.2))
    offsets = {e: (i - (len(ENCODERS) - 1) / 2) * 0.09 for i, e in enumerate(ENCODERS)}
    for encoder in ENCODERS:
        ys = [i + offsets[encoder] for i, d in enumerate(order) if (encoder, d) in loss]
        xs = [loss[(encoder, d)] for d in order if (encoder, d) in loss]
        marker = "o" if "single" in encoder else "^"
        axis.scatter(xs, ys, s=46, marker=marker, color=colour_of(encoder), edgecolor="white",
                     linewidth=1.2, zorder=3, label=encoder)
    axis.axvline(1.0, color="#52514e", linewidth=1, linestyle="--")
    axis.set_yticks(range(len(order)), order, fontsize=8)
    axis.set_xlabel("normalized VALIDATION loss of a fresh head (1.0 = mean predictor; lower is better)",
                    fontsize=9, color=INK)
    axis.set_title("Held-out dojos: fresh heads on each frozen encoder (circles single-card, triangles multi-card)",
                   fontsize=10, color=INK)
    axis.grid(axis="x", color=GRID)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    axis.legend(fontsize=8, frameon=False, loc="lower right")
    figure.tight_layout()
    figure.savefig(RESULTS / "extrinsic_by_dojo.png", dpi=130)
    plt.close(figure)


if __name__ == "__main__":
    main()
