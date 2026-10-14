"""Tables and figures from intrinsic_suite.py's outputs.

Reads data/evaluations/2026-10-11_model_by_diet_r2/intrinsic/<encoder>/
<label source>/<analysis>/ (scalars.json, projection.csv) and writes:

    results/intrinsic_scalars.csv        every scalar, long format: encoder,
                                         label_source, analysis, scalar, value
    results/intrinsic_summary.csv        one row per (encoder, label source):
                                         ARI, NMI, silhouette, nearest-centroid
                                         accuracy and its majority baseline;
                                         label source "unlabeled" carries the
                                         effective-rank scalars
    results/intrinsic_by_label.png       nearest-centroid accuracy and NMI per
                                         label source: the three diets against
                                         the range of the three controls, per
                                         model shape
    results/intrinsic_projection_<source>.png
                                         t-SNE contact sheets (one panel per
                                         encoder, controls seed 0), colored by
                                         label, from projection.csv
    results/learning_curves/<dojo>.png   copies of plot_learning_curves' charts

Run (from the project root):
    python experiments/2026-10-11_model_by_diet_r2/analysis/intrinsic_report.py
"""

import csv
import json
import shutil
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

EVAL = Path("data/evaluations/2026-10-11_model_by_diet_r2")
RESULTS = Path("experiments/2026-10-11_model_by_diet_r2/results")
DIETS = ("contrastive", "metric", "mixed")
DIET_COLOURS = {"contrastive": "#2a78d6", "metric": "#eb6834", "mixed": "#1baf7a"}
CONTROL_COLOUR = "#b9b8b2"
INK = "#2b2a27"
GRID = "#e5e4e0"
# Up to eight labels colored, the rest folded into "other"
LABEL_COLOURS = ("#2a78d6", "#eb6834", "#1baf7a", "#d63a7e", "#8a5cd1", "#c9a227", "#3fb6c9", "#7a4a2a")
SUMMARY_SCALARS = (
    "ari", "nmi", "silhouette", "nearest_centroid_accuracy", "majority_baseline",
    "effective_rank", "effective_rank_fraction", "top1_variance_share", "top5_variance_share",
    "n_cards", "n_labels",
)
PROJECTION_SOURCES = ("game", "heldout_hs_class", "trained_mtg_card_type", "holdout_tier")


def read_scalars() -> list[dict]:
    rows = []
    for path in sorted((EVAL / "intrinsic").glob("*/*/*/scalars.json")):
        analysis, source, encoder = path.parent.name, path.parent.parent.name, path.parents[2].name
        for scalar, value in json.loads(path.read_text()).items():
            rows.append({"encoder": encoder, "label_source": source, "analysis": analysis,
                         "scalar": scalar, "value": value})
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> list[dict]:
    wide: dict[tuple[str, str], dict] = {}
    for row in rows:
        key = (row["encoder"], row["label_source"])
        entry = wide.setdefault(key, {"encoder": key[0], "label_source": key[1]})
        if row["scalar"] in SUMMARY_SCALARS:
            entry[row["scalar"]] = round(row["value"], 4)
    return [{"encoder": e["encoder"], "label_source": e["label_source"],
             **{s: e.get(s, "") for s in SUMMARY_SCALARS}} for e in wide.values()]


def plot_by_label(summary: list[dict]) -> None:
    value = {(r["encoder"], r["label_source"], s): r[s] for r in summary
             for s in ("nearest_centroid_accuracy", "nmi") if r[s] != ""}
    sources = sorted({r["label_source"] for r in summary if r["label_source"] != "unlabeled"})
    figure, axes = plt.subplots(2, 2, figsize=(12, 7.5), sharey=True)
    for col, scalar in enumerate(("nearest_centroid_accuracy", "nmi")):
        for row, shape in enumerate(("single", "multi")):
            axis = axes[row][col]
            for y, source in enumerate(sources):
                controls = [value[(f"control_{shape}_s{s}", source, scalar)] for s in (0, 1, 2)
                            if (f"control_{shape}_s{s}", source, scalar) in value]
                if controls:
                    axis.plot([min(controls), max(controls)], [y, y], color=CONTROL_COLOUR,
                              linewidth=7, solid_capstyle="round", zorder=1)
                for diet in DIETS:
                    key = (f"{shape}_{diet}", source, scalar)
                    if key in value:
                        axis.scatter(value[key], y, s=42, color=DIET_COLOURS[diet], edgecolor="white",
                                     linewidth=1.1, zorder=3, label=diet if y == 0 else None)
            if scalar == "nearest_centroid_accuracy":
                for y, source in enumerate(sources):
                    key = (f"control_{shape}_s0", source)
                    base = next((r["majority_baseline"] for r in summary
                                 if (r["encoder"], r["label_source"]) == key), "")
                    if base != "":
                        axis.plot([base], [y], marker="|", markersize=12, color=INK, zorder=2)
            axis.set_yticks(range(len(sources)), sources, fontsize=8)
            axis.set_title(f"{shape}-card: {scalar.replace('_', ' ')}", fontsize=10, color=INK)
            axis.grid(axis="x", color=GRID)
            for side in ("top", "right"):
                axis.spines[side].set_visible(False)
    axes[0][0].legend(fontsize=8, frameon=False, loc="lower right")
    figure.suptitle("Intrinsic analyses per label source (grey bar = range of 3 untrained controls; "
                    "| = majority baseline)", fontsize=10, color=INK)
    figure.tight_layout()
    figure.savefig(RESULTS / "intrinsic_by_label.png", dpi=130)
    plt.close(figure)


def plot_projections(source: str) -> None:
    encoders = ("control_single_s0", "single_contrastive", "single_metric", "single_mixed",
                "control_multi_s0", "multi_contrastive", "multi_metric", "multi_mixed")
    tables = {}
    for encoder in encoders:
        path = EVAL / "intrinsic" / encoder / source / "projection" / "projection.csv"
        if path.exists():
            with path.open(encoding="utf-8", newline="") as handle:
                tables[encoder] = list(csv.DictReader(handle))
    if not tables:
        return
    counts = Counter(r["label"] for r in next(iter(tables.values())))
    shown = [label for label, _ in counts.most_common(len(LABEL_COLOURS))]
    figure, axes = plt.subplots(2, 4, figsize=(14, 7.4))
    for axis, encoder in zip(axes.flat, encoders):
        axis.set_xticks([])
        axis.set_yticks([])
        for side in axis.spines.values():
            side.set_color(GRID)
        axis.set_title(encoder, fontsize=9, color=INK)
        rows = tables.get(encoder, [])
        other = [r for r in rows if r["label"] not in shown]
        axis.scatter([float(r["x"]) for r in other], [float(r["y"]) for r in other], s=2,
                     color=CONTROL_COLOUR, linewidth=0)
        for label, colour in zip(shown, LABEL_COLOURS):
            mine = [r for r in rows if r["label"] == label]
            axis.scatter([float(r["x"]) for r in mine], [float(r["y"]) for r in mine], s=2,
                         color=colour, linewidth=0, label=label)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=min(8, len(labels)), fontsize=8,
                  frameon=False, markerscale=5)
    figure.suptitle(f"t-SNE of each encoder's embeddings, colored by {source} "
                    f"(PerLabelCap 500, seed 0)", fontsize=10, color=INK)
    figure.tight_layout(rect=(0, 0.05, 1, 0.97))
    figure.savefig(RESULTS / f"intrinsic_projection_{source}.png", dpi=120)
    plt.close(figure)


def main() -> None:
    rows = read_scalars()
    write_csv(RESULTS / "intrinsic_scalars.csv", rows)
    summary = summarize(rows)
    write_csv(RESULTS / "intrinsic_summary.csv", summary)
    plot_by_label(summary)
    for source in PROJECTION_SOURCES:
        plot_projections(source)
    curves = RESULTS / "learning_curves"
    curves.mkdir(exist_ok=True)
    for png in (EVAL / "extrinsic" / "plots").glob("*.png"):
        shutil.copy(png, curves / png.name)
    print(f"{len(rows)} scalars, {len(summary)} summary rows")


if __name__ == "__main__":
    main()
