"""Per-round comparison of the two arms (Method, steps 3-4): the quality
and speed verdict inputs, as tables and one figure.

For each round k, compare.py's paired, deck-stratified bootstrap of
T_rk - R_rk (diet-weighted deck AUC), plus each arm's absolute scores
from results/eval_summary.csv. Applies the Aim's speed rule (every
first-half round: delta >= +0.02 supported; any <= -0.02 refuted) and
quality rule (final round, via compare.quality_rule). Rules are applied
to the point delta and CI exactly as written; interpretation is the
PI's.

Writes results/round_compare.csv and results/deck_auc_by_round.png.
Run (project root, PYTHONPATH=.):
    python experiments/2026-11-11_teacher_vs_real_pairs/analysis/round_curves.py
"""

import csv
import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ANALYSIS = Path(__file__).parent
RESULTS = ANALYSIS.parent / "results"
ROUNDS = range(8)
STEPS_PER_ROUND = 10_000
FIRST_HALF = range(4)
MARGIN = 0.02
# Reference palette slots 1 and 2 (dataviz palette.md), fixed order: R, T
COLOR = {"R": "#2a78d6", "T": "#eb6834"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _load_compare():
    spec = importlib.util.spec_from_file_location("compare", ANALYSIS / "compare.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def summary_rows() -> dict[str, dict]:
    with (RESULTS / "eval_summary.csv").open(encoding="utf-8", newline="") as f:
        return {r["encoder"]: r for r in csv.DictReader(f)}


def main() -> None:
    compare = _load_compare()
    summary = summary_rows()
    rows = []
    for k in ROUNDS:
        a, b = f"R_r{k}", f"T_r{k}"
        diffs = compare.paired_by_game(compare.per_deck(a), compare.per_deck(b))
        delta = compare.weighted({g: float(d.mean()) for g, d in diffs.items()})
        lo, hi = compare.bootstrap_ci(diffs)
        rows.append({
            "round": k, "step": (k + 1) * STEPS_PER_ROUND,
            "R_deck_auc": float(summary[a]["deck_auc_weighted"]),
            "T_deck_auc": float(summary[b]["deck_auc_weighted"]),
            "delta": delta, "lo95": lo, "hi95": hi,
            "R_gwent": float(summary[a]["deck_auc_gwent"]), "T_gwent": float(summary[b]["deck_auc_gwent"]),
            "R_linear_cls": float(summary[a]["linear_cls"]), "T_linear_cls": float(summary[b]["linear_cls"]),
            "R_eff_rank": float(summary[a]["effective_rank"]), "T_eff_rank": float(summary[b]["effective_rank"]),
        })
    with (RESULTS / "round_compare.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    first = [r["delta"] for r in rows if r["round"] in FIRST_HALF]
    speed = ("refuted" if min(first) <= -MARGIN
             else "supported" if all(d >= MARGIN for d in first) else "no clear difference")
    last = rows[-1]
    cls_delta = last["T_linear_cls"] - last["R_linear_cls"]
    quality = compare.quality_rule(last["delta"], last["lo95"], last["hi95"], cls_delta)
    for r in rows:
        print(f"r{r['round']} step {r['step']:>6}: R {r['R_deck_auc']:.4f}  T {r['T_deck_auc']:.4f}  "
              f"delta {r['delta']:+.4f} [{r['lo95']:+.4f}, {r['hi95']:+.4f}]  "
              f"cls R {r['R_linear_cls']:.3f} T {r['T_linear_cls']:.3f}")
    print(f"speed rule (first-half rounds {list(FIRST_HALF)}): {speed}")
    print(f"quality rule (final round, linear-cls delta {cls_delta:+.4f}): {quality}")

    steps = [r["step"] / 1000 for r in rows]
    fig, (top, bottom, probe) = plt.subplots(3, 1, figsize=(7.5, 9.5), sharex=True,
                                             gridspec_kw={"height_ratios": [3, 2, 2]})
    # End labels: R above its last point, T below, so close finishes don't collide
    offset = {"R": 7, "T": -9}
    for arm, label in (("R", "R: real-deck pairs"), ("T", "T: teacher-neighbour pairs")):
        values = [r[f"{arm}_deck_auc"] for r in rows]
        top.plot(steps, values, color=COLOR[arm], linewidth=2, marker="o", markersize=5, label=label)
        top.annotate(f"{label.split(':')[0]} {values[-1]:.3f}", (steps[-1], values[-1]),
                     xytext=(6, offset[arm]), textcoords="offset points", va="center", color=INK, fontsize=9)
    for name, text in (("frozen_mean", "frozen whitened mean (no head)"), ("untrained", "untrained head")):
        if name in summary:
            y = float(summary[name]["deck_auc_weighted"])
            top.axhline(y, color=MUTED, linewidth=1, linestyle="--")
            top.annotate(text, (steps[0], y), xytext=(0, 4), textcoords="offset points",
                         color=MUTED, fontsize=8)
    top.set_ylabel("diet-weighted same-deck AUC", color=INK)
    top.set_title("Same-deck retrieval AUC by training step (VALIDATION decks)", color=INK, fontsize=11, loc="left")
    top.legend(frameon=False, loc="center right", fontsize=9)

    deltas = [r["delta"] for r in rows]
    bottom.fill_between(steps, [r["lo95"] for r in rows], [r["hi95"] for r in rows],
                        color=COLOR["T"], alpha=0.2, linewidth=0)
    bottom.plot(steps, deltas, color=COLOR["T"], linewidth=2, marker="o", markersize=5)
    bottom.axhline(0, color=MUTED, linewidth=1)
    for y in (MARGIN, -MARGIN):
        bottom.axhline(y, color=MUTED, linewidth=1, linestyle=":")
    bottom.annotate("±0.02 margin", (steps[0], MARGIN), xytext=(0, 4), textcoords="offset points",
                    color=MUTED, fontsize=8)
    bottom.axvspan(steps[0] - 2, steps[FIRST_HALF[-1]] + 2, color=GRID, alpha=0.5, linewidth=0)
    bottom.set_ylabel("T − R (paired, 95% CI)", color=INK)
    bottom.set_xlabel("training steps (thousands)", color=INK)
    bottom.set_title("Difference; shaded: first-half rounds (speed rule)", color=INK, fontsize=10, loc="left")
    for arm in ("R", "T"):
        probe.plot(steps, [r[f"{arm}_linear_cls"] for r in rows], color=COLOR[arm], linewidth=2,
                   marker="o", markersize=5)
        probe.annotate(arm, (steps[-1], rows[-1][f"{arm}_linear_cls"]), xytext=(6, 0),
                       textcoords="offset points", va="center", color=INK, fontsize=9)
    for name, text in (("frozen_mean", "frozen whitened mean"), ("untrained", "untrained head")):
        if name in summary:
            y = float(summary[name]["linear_cls"])
            probe.axhline(y, color=MUTED, linewidth=1, linestyle="--")
            probe.annotate(text, (steps[0], y), xytext=(0, -11), textcoords="offset points",
                           color=MUTED, fontsize=8)
    probe.set_ylabel("linear-cls (balanced acc.)", color=INK)
    probe.set_title("Card-field probe (quality guard: T may not drop > 0.02 below R)", color=INK,
                    fontsize=10, loc="left")
    probe.set_xlabel("training steps (thousands)", color=INK)
    bottom.set_xlabel("")
    for ax in (top, bottom, probe):
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color(MUTED)
        ax.spines["bottom"].set_color(MUTED)
        ax.tick_params(colors=MUTED)
    fig.tight_layout()
    fig.savefig(RESULTS / "deck_auc_by_round.png", dpi=150, facecolor="#fcfcfb")
    print(RESULTS / "deck_auc_by_round.png")


if __name__ == "__main__":
    main()
