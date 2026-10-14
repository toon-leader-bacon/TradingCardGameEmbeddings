"""Tables and figures for the held-out game: Gwent fresh heads and zero-shot
contrastive scores.

Reads results/extrinsic_gwent_validation.csv (extrinsic_gwent.py) and
results/zero_shot_contrastive.csv (zero_shot_contrastive.py) and writes:

    results/extrinsic_gwent_summary.csv   per encoder: median normalized
                                          VALIDATION loss over the Gwent dojos,
                                          and the median paired difference
                                          against the mean of its shape's
                                          three controls
    results/extrinsic_gwent_by_dojo.png   per Gwent dojo: the three diets
                                          against the range of the three
                                          controls, per model shape
    results/zero_shot_contrastive_summary.csv
                                          per encoder and game: median
                                          normalized contrastive VALIDATION
                                          loss over the styles (1.0 = chance)
    results/zero_shot_contrastive_by_game.png
                                          the same, per game

Lower is better everywhere (1.0 = mean predictor / prior / chance).

Run (from the project root):
    python experiments/2026-10-11_model_by_diet_r2/analysis/gwent_report.py
"""

import csv
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = Path("experiments/2026-10-11_model_by_diet_r2/results")
DIETS = ("contrastive", "metric", "mixed")
SHAPES = ("single", "multi")
SEEDS = (0, 1, 2)
DIET_COLOURS = {"contrastive": "#2a78d6", "metric": "#eb6834", "mixed": "#1baf7a"}
CONTROL_COLOUR = "#b9b8b2"
INK = "#2b2a27"
GRID = "#e5e4e0"


def read(name: str) -> list[dict]:
    with (RESULTS / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write(name: str, rows: list[dict]) -> None:
    with (RESULTS / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def dot_panels(value: dict, keys: list[str], title: str, xlabel: str, path: Path) -> None:
    """value[(encoder, key)] -> number; one panel per model shape."""
    figure, axes = plt.subplots(1, 2, figsize=(12, 0.42 * len(keys) + 1.8), sharey=True)
    for axis, shape in zip(axes, SHAPES):
        for y, key in enumerate(keys):
            controls = [value[(f"control_{shape}_s{s}", key)] for s in SEEDS
                        if (f"control_{shape}_s{s}", key) in value]
            if controls:
                axis.plot([min(controls), max(controls)], [y, y], color=CONTROL_COLOUR, linewidth=7,
                          solid_capstyle="round", zorder=1, label="controls (3 seeds)" if y == 0 else None)
            for diet in DIETS:
                if (f"{shape}_{diet}", key) in value:
                    axis.scatter(value[(f"{shape}_{diet}", key)], y, s=42, color=DIET_COLOURS[diet],
                                 edgecolor="white", linewidth=1.1, zorder=3, label=diet if y == 0 else None)
        axis.axvline(1.0, color="#52514e", linewidth=1, linestyle="--")
        axis.set_yticks(range(len(keys)), keys, fontsize=8)
        axis.set_title(f"{shape}-card", fontsize=10, color=INK)
        axis.set_xlabel(xlabel, fontsize=9, color=INK)
        axis.grid(axis="x", color=GRID)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
    axes[0].legend(fontsize=8, frameon=False, loc="best")
    figure.suptitle(title, fontsize=10, color=INK)
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


def extrinsic() -> None:
    rows = read("extrinsic_gwent_validation.csv")
    loss = {(r["encoder"], r["dojo"]): float(r["validation_normalized"]) for r in rows}
    dojos = sorted({r["dojo"] for r in rows})
    encoders = sorted({r["encoder"] for r in rows})
    summary = []
    for encoder in encoders:
        shape = "single" if "single" in encoder else "multi"
        diffs = []
        for dojo in dojos:
            controls = [loss[(f"control_{shape}_s{s}", dojo)] for s in SEEDS
                        if (f"control_{shape}_s{s}", dojo) in loss]
            if (encoder, dojo) in loss and controls:
                diffs.append(loss[(encoder, dojo)] - statistics.mean(controls))
        values = [loss[(encoder, d)] for d in dojos if (encoder, d) in loss]
        summary.append({
            "encoder": encoder,
            "n_dojos": len(values),
            "median_normalized": round(statistics.median(values), 4),
            "n_below_1": sum(v < 1 for v in values),
            "median_minus_control_mean": round(statistics.median(diffs), 4),
            "n_better_than_control_mean_by_0.02": sum(d < -0.02 for d in diffs),
            "n_worse_than_control_mean_by_0.02": sum(d > 0.02 for d in diffs),
        })
    write("extrinsic_gwent_summary.csv", summary)
    for row in summary:
        print(row)
    dot_panels(loss, dojos, "Gwent (held-out game): fresh heads on each frozen encoder",
               "normalized VALIDATION loss (1.0 = mean predictor; lower is better)",
               RESULTS / "extrinsic_gwent_by_dojo.png")


def zero_shot() -> None:
    rows = read("zero_shot_contrastive.csv")
    by_game: dict[tuple[str, str], list[float]] = {}
    trained = {}
    for r in rows:
        by_game.setdefault((r["encoder"], r["game"]), []).append(float(r["normalized"]))
        trained[r["game"]] = r["game_trained"]
    value = {key: statistics.median(v) for key, v in by_game.items()}
    summary = [{"encoder": e, "game": g, "game_trained": trained[g], "n_styles": len(by_game[(e, g)]),
                "median_normalized": round(v, 4)} for (e, g), v in sorted(value.items())]
    write("zero_shot_contrastive_summary.csv", summary)
    games = sorted({g for _, g in value}, key=lambda g: (trained[g] == "True", g))
    labels = [f"{g} ({'trained' if trained[g] == 'True' else 'unseen'})" for g in games]
    relabeled = {(e, labels[games.index(g)]): v for (e, g), v in value.items()}
    dot_panels(relabeled, labels,
               "Zero-shot contrastive VALIDATION loss per game (median over styles; 1.0 = chance)",
               "normalized contrastive loss (lower is better)", RESULTS / "zero_shot_contrastive_by_game.png")
    for game in games:
        print(game, {e: round(value[(e, game)], 3) for e, g in sorted(value) if g == game})


def main() -> None:
    if (RESULTS / "extrinsic_gwent_validation.csv").exists():
        extrinsic()
    if (RESULTS / "zero_shot_contrastive.csv").exists():
        zero_shot()


if __name__ == "__main__":
    main()
