"""Tables and figures from the probe (probe_cells.py's CSVs).

Reads results/probe_summary.csv, probe_by_field.csv, probe_deck_by_game.csv
and writes into results/:

    probe_vs_control.csv      each cell's isolated-view deck AUC and
                              linear-cls against its model shape's
                              untrained controls (seed 0, and the mean of
                              seeds 0-2), on the full probe and on the
                              out-of-sample slice, with the Aim's bar
    probe_tradeoff.png        deck AUC against linear-cls, every model
    probe_effective_rank.png  effective rank of the isolated embeddings
    probe_deck_by_game.png    deck AUC per game (Gwent is the unseen game)
    probe_held_out_fields.png linear-cls on non-TRAIN cards, held-out fields

Run (from the project root):
    python experiments/2026-10-11_model_by_diet_r2/analysis/probe_report.py
"""

import csv
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = Path("experiments/2026-10-11_model_by_diet_r2/results")
BAR_DECK = 0.03
BAR_CLS = 0.03
DIETS = ("contrastive", "metric", "mixed")
SHAPES = ("single", "multi")
# Validated default categorical palette, slots 1-3, one per diet
DIET_COLOURS = {"contrastive": "#2a78d6", "metric": "#eb6834", "mixed": "#1baf7a"}
SHAPE_MARKERS = {"single": "o", "multi": "^"}
CONTROL_COLOUR = "#8a8984"
REFERENCE_COLOUR = "#52514e"
INK = "#2b2a27"
GRID = "#e5e4e0"


def read(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def isolated(summary: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {r["model"]: r for r in summary if r["view"] in ("isolated", "frozen")}


def style_axis(axis) -> None:
    axis.grid(color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    axis.tick_params(colors=INK, labelsize=9)


def vs_control_rows(rows: dict[str, dict[str, str]]) -> list[dict[str, object]]:
    out = []
    for shape in SHAPES:
        controls = [rows[f"r2_control_{shape}_s{s}"] for s in (0, 1, 2)]

        def mean(key: str) -> float:
            return statistics.mean(float(c[key]) for c in controls)

        for diet in DIETS:
            cell = rows[f"r2_{shape}_{diet}"]
            entry: dict[str, object] = {"cell": f"{shape}_{diet}"}
            for label, key in (
                ("deck_auc", "deck_auc"),
                ("linear_cls", "linear_cls_bacc"),
                ("gwent_deck_auc", "deck_auc_gwent"),
                ("slice_linear_cls", "slice_linear_cls_bacc"),
            ):
                value = float(cell[key])
                entry[label] = round(value, 4)
                entry[f"{label}_minus_control_s0"] = round(value - float(controls[0][key]), 4)
                entry[f"{label}_minus_control_mean"] = round(value - mean(key), 4)
            for scope, deck, cls in (
                ("full", "deck_auc", "linear_cls"),
                ("slice", "gwent_deck_auc", "slice_linear_cls"),
            ):
                for ref in ("s0", "mean"):
                    passed = (
                        float(entry[f"{deck}_minus_control_{ref}"]) >= BAR_DECK
                        and float(entry[f"{cls}_minus_control_{ref}"]) >= -BAR_CLS
                    )
                    entry[f"bar_{scope}_vs_{ref}"] = "met" if passed else "not met"
            out.append(entry)
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_tradeoff(rows: dict[str, dict[str, str]], path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    panels = (
        ("full probe", "deck_auc", "linear_cls_bacc", "deck AUC (0.5 = chance)",
         "linear-cls balanced accuracy"),
        ("out-of-sample slice", "deck_auc_gwent", "slice_linear_cls_bacc",
         "Gwent deck AUC (unseen game)", "linear-cls, held-out fields, non-TRAIN cards"),
    )
    for axis, (title, x_key, y_key, x_label, y_label) in zip(axes, panels):
        for shape in SHAPES:
            xs = [float(rows[f"r2_control_{shape}_s{s}"][x_key]) for s in (0, 1, 2)]
            ys = [float(rows[f"r2_control_{shape}_s{s}"][y_key]) for s in (0, 1, 2)]
            axis.scatter(xs, ys, s=64, marker=SHAPE_MARKERS[shape], color=CONTROL_COLOUR,
                         edgecolor="white", linewidth=1.5, zorder=3,
                         label=f"untrained {shape}-card (3 seeds)")
            for diet in DIETS:
                row = rows[f"r2_{shape}_{diet}"]
                x, y = float(row[x_key]), float(row[y_key])
                axis.scatter([x], [y], s=90, marker=SHAPE_MARKERS[shape],
                             color=DIET_COLOURS[diet], edgecolor="white", linewidth=1.5,
                             zorder=4, label=f"{shape}_{diet}")
                axis.annotate(f"{shape}_{diet}", (x, y), xytext=(6, 4) if shape == "single" else (6, -12),
                              textcoords="offset points", fontsize=8, color=INK)
        reference = rows["r2_reference_whitened_mean"]
        axis.scatter([float(reference[x_key])], [float(reference[y_key])], s=110, marker="*",
                     color=REFERENCE_COLOUR, edgecolor="white", linewidth=1.5, zorder=4,
                     label="frozen whitened mean (768-d, no head)")
        axis.axvline(0.5, color=REFERENCE_COLOUR, linewidth=1, linestyle="--")
        axis.set_title(title, fontsize=10, color=INK)
        axis.set_xlabel(x_label, fontsize=9, color=INK)
        axis.set_ylabel(y_label, fontsize=9, color=INK)
        style_axis(axis)
    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    figure.legend(unique.values(), unique.keys(), loc="lower center", ncol=5, fontsize=8,
                  frameon=False)
    figure.tight_layout(rect=(0, 0.12, 1, 1))
    figure.savefig(path, dpi=130)
    plt.close(figure)


def plot_effective_rank(rows: dict[str, dict[str, str]], path: Path) -> None:
    order = [f"r2_{s}_{d}" for s in SHAPES for d in DIETS]
    order += [f"r2_control_{s}_s0" for s in SHAPES] + ["r2_reference_whitened_mean"]
    colours = [DIET_COLOURS[m.split("_")[-1]] for m in order[:6]]
    colours += [CONTROL_COLOUR, CONTROL_COLOUR, REFERENCE_COLOUR]
    labels = [m.removeprefix("r2_").replace("reference_whitened_mean", "frozen whitened mean")
              for m in order]
    values = [float(rows[m]["effective_rank"]) for m in order]
    figure, axis = plt.subplots(figsize=(9, 3.8))
    bars = axis.bar(range(len(order)), values, color=colours, edgecolor="white", linewidth=2)
    for bar, value in zip(bars, values):
        axis.annotate(f"{value:.1f}", (bar.get_x() + bar.get_width() / 2, value),
                      xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8,
                      color=INK)
    axis.set_yscale("log")
    axis.set_xticks(range(len(order)), labels, rotation=30, ha="right", fontsize=8)
    axis.set_ylabel("effective rank (log scale)", fontsize=9, color=INK)
    axis.set_title("Effective rank of the isolated card embeddings (256-d; frozen: 768-d)",
                   fontsize=10, color=INK)
    style_axis(axis)
    axis.grid(axis="x", visible=False)
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


def plot_by_game(by_game: list[dict[str, str]], path: Path) -> None:
    games = ["mtg", "pokemon", "slay_the_spire_2", "flesh_and_blood", "dominion", "gwent"]
    models = [f"r2_{s}_{d}" for s in SHAPES for d in DIETS]
    table = {(r["model"], r["game"]): float(r["deck_auc"]) for r in by_game
             if r["view"] in ("isolated", "frozen")}
    figure, axes = plt.subplots(1, len(games), figsize=(13, 3.6), sharey=True)
    for axis, game in zip(axes, games):
        values = [table[(m, game)] for m in models]
        colours = [DIET_COLOURS[m.split("_")[-1]] for m in models]
        hatches = ["" if m.startswith("r2_single") else "///" for m in models]
        bars = axis.bar(range(len(models)), values, color=colours, edgecolor="white",
                        linewidth=1.5)
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)
        control = statistics.mean(table[(f"r2_control_{s}_s{k}", game)]
                                  for s in SHAPES for k in (0, 1, 2))
        axis.axhline(control, color=CONTROL_COLOUR, linewidth=1.5, label="untrained (mean)")
        axis.axhline(table[("r2_reference_whitened_mean", game)], color=REFERENCE_COLOUR,
                     linewidth=1.5, linestyle=":", label="frozen whitened mean")
        axis.axhline(0.5, color=REFERENCE_COLOUR, linewidth=0.8, linestyle="--")
        axis.set_title(game + (" (unseen)" if game == "gwent" else ""), fontsize=9, color=INK)
        axis.set_xticks([])
        axis.set_ylim(0.35, 1.0)
        style_axis(axis)
        axis.grid(axis="x", visible=False)
    axes[0].set_ylabel("deck AUC (0.5 = chance)", fontsize=9, color=INK)
    from matplotlib.patches import Patch

    handles = [Patch(color=DIET_COLOURS[d], label=d) for d in DIETS]
    handles += [Patch(facecolor="white", edgecolor=INK, label="single-card (solid)"),
                Patch(facecolor="white", edgecolor=INK, hatch="///", label="multi-card (hatched)")]
    line_handles, line_labels = axes[0].get_legend_handles_labels()
    figure.legend(handles + line_handles[:2], [h.get_label() for h in handles] + line_labels[:2],
                  loc="lower center", ncol=7, fontsize=8, frameon=False)
    figure.tight_layout(rect=(0, 0.1, 1, 1))
    figure.savefig(path, dpi=130)
    plt.close(figure)


def plot_held_out_fields(by_field: list[dict[str, str]], path: Path) -> None:
    fields = ["hearthstone.class", "flesh_and_blood.pitch", "slay_the_spire_2.type"]
    models = [f"r2_{s}_{d}" for s in SHAPES for d in DIETS]
    table = {(r["model"], r["field"]): float(r["linear"]) for r in by_field
             if r["probe_set"].startswith("slice") and r["view"] in ("isolated", "frozen")}
    figure, axes = plt.subplots(1, len(fields), figsize=(11, 3.6), sharey=True)
    for axis, field in zip(axes, fields):
        values = [table[(m, field)] for m in models]
        bars = axis.bar(range(len(models)), values,
                        color=[DIET_COLOURS[m.split("_")[-1]] for m in models],
                        edgecolor="white", linewidth=1.5)
        for bar, model in zip(bars, models):
            bar.set_hatch("" if model.startswith("r2_single") else "///")
        control = statistics.mean(table[(f"r2_control_{s}_s{k}", field)]
                                  for s in SHAPES for k in (0, 1, 2))
        axis.axhline(control, color=CONTROL_COLOUR, linewidth=1.5)
        axis.axhline(table[("r2_reference_whitened_mean", field)], color=REFERENCE_COLOUR,
                     linewidth=1.5, linestyle=":")
        axis.set_title(field, fontsize=9, color=INK)
        axis.set_xticks([])
        axis.set_ylim(0, 1.0)
        style_axis(axis)
        axis.grid(axis="x", visible=False)
    axes[0].set_ylabel("linear-cls balanced accuracy\n(non-TRAIN cards; no dojo trains these)",
                       fontsize=9, color=INK)
    figure.suptitle("Grey line: untrained (mean of 6 controls); dotted: frozen whitened mean. "
                    "Colour = diet; hatched = multi-card.", fontsize=9, color=INK)
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


def main() -> None:
    summary = read("probe_summary.csv")
    rows = isolated(summary)
    table = vs_control_rows(rows)
    write_csv(RESULTS / "probe_vs_control.csv", table)
    for row in table:
        print(row)
    plot_tradeoff(rows, RESULTS / "probe_tradeoff.png")
    plot_effective_rank(rows, RESULTS / "probe_effective_rank.png")
    plot_by_game(read("probe_deck_by_game.csv"), RESULTS / "probe_deck_by_game.png")
    plot_held_out_fields(read("probe_by_field.csv"), RESULTS / "probe_held_out_fields.png")


if __name__ == "__main__":
    main()
