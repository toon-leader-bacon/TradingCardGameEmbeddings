"""Summarize every finished run's --profile CSVs and Windows counter
samples into results/.

Reads, per run, under data/runs/2026-10-24_scale_profile/:
    <run>/profile/{steps,builds,reserves,evaluation,marks,samples}.csv
    logs/<run>.{resources,utilization}.csv     (Windows counter samplers)

Writes into experiments/2026-10-24_scale_profile/results/:
    run_summary.csv     one row per run: startup stages, training time,
                        throughput, step time and its parts (shares of
                        summed step time), cache hit rates, evictions,
                        memory, GPU and CPU use, TEST pass time
    hourly.csv          per run and hour of training: cards/s, step parts,
                        cache hit rate, VRAM, GPU use
    by_family.csv       per run and dojo family (contrastive style; metric
                        source): steps, time share, step time, cards/s
    by_dojo.csv         per run and dojo: steps, share of training time,
                        median step and wait, cards per step, median build
                        time per card, live texts per step, passes started
    test_by_dojo.csv    per run and dojo: mean TEST pass seconds per round
    mtg_projection.csv  per mtg_* run: steady-state anchor cards/s and the
                        hours one anchor pass over every known card of every
                        TRAIN MTG deck would take
    step_parts.png, hourly_throughput.png, timeline.png (5-minute bins:
    cards/s, wait share and texts encoded, per run)

Steady state is the second half of a run's training time (cache warm,
first passes over small datasets done). Hit rate is 1 - encoded / texts
asked between two samples (FeatureCacheStats; the warm-up's encodes
happen before training and are excluded by differencing); it includes
TEST passes and task-mod texts. The Windows counter samples (GPU, CPU,
RAM) are clipped to the same steady window, by wall time (marks.csv
maps the profile clock t to wall time). Rates are given two ways:
cards per second of summed step time, and wall-clock cards per second
over the steady window (TEST, checkpoints and gaps included), which is
what a projection of a long run uses.

Usage (from the project root):
    python experiments/2026-10-24_scale_profile/analysis/profile_report.py [run ...]
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

RUNS = Path("data/runs/2026-10-24_scale_profile")
RESULTS = Path("experiments/2026-10-24_scale_profile/results")
DEFAULT_RUNS = ["pilot", "mixed_n0", "mixed_n4", "mtg_n0", "mtg_n4"]
STAGES = ["mods_s", "serialization_s", "cache_lookup_s", "live_encoding_s"]
# MTG deck box (data/final/decks/mtg.db), counted 2026-10-24 by a one-off
# scan (data/runs/2026-10-24_scale_profile/mtg_deck_card_totals.txt)
MTG_DECKS = 6_911_380
MTG_DECK_CARD_ROWS = 161_289_978  # distinct cards per deck, summed
MTG_TRAIN_SHARE = 0.8  # deck split 8/1/1
# Share of a TRAIN deck's rows that are TRAIN-visible known cards: 0.8 (the
# card tiers) until analysis/mtg_train_visits.py measures it
MTG_CARD_FACTOR_FILE = RUNS / "mtg_train_card_factor.txt"


def read(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


def family(dojo: str) -> str:
    source = dojo.split(".", 1)[0]
    return source if source.startswith("contrastive") else f"metric:{source}"


def marks_durations(marks: pd.DataFrame) -> dict[str, float]:
    """Seconds of each "<x> start" / "<x> end" pair in marks.csv."""
    result: dict[str, float] = {}
    starts = {m[: -len(" start")]: t for m, t in zip(marks["mark"], marks["t"]) if m.endswith(" start")}
    for mark, t in zip(marks["mark"], marks["t"]):
        if mark.endswith(" end"):
            name = mark[: -len(" end")]
            if name in starts:
                result[name] = t - starts[name]
    return result


def wall_of(marks: pd.DataFrame, t: float) -> pd.Timestamp:
    """The wall time of profile clock t (marks.csv's first row anchors both)."""
    first = marks.iloc[0]
    return pd.Timestamp(first["wall"]) + pd.Timedelta(seconds=t - first["t"])


def clipped(samples: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Windows counter rows whose time falls in [start, end]."""
    if samples.empty:
        return samples
    times = pd.to_datetime(samples["time"])
    return samples[(times >= start) & (times <= end)]


def hit_rate(samples: pd.DataFrame) -> float:
    if len(samples) < 2:
        return float("nan")
    asked = samples["cache_texts"].iloc[-1] - samples["cache_texts"].iloc[0]
    encoded = samples["cache_encoded"].iloc[-1] - samples["cache_encoded"].iloc[0]
    return float("nan") if asked <= 0 else 1 - encoded / asked


def training_window(steps: pd.DataFrame) -> tuple[float, float]:
    return float(steps["t"].min() - steps["total_s"].iloc[0]), float(steps["t"].max())


def summarize(run: str) -> dict | None:
    profile = RUNS / run / "profile"
    steps = read(profile / "steps.csv")
    if steps.empty:
        return None
    marks = read(profile / "marks.csv")
    samples = read(profile / "samples.csv")
    evaluation = read(profile / "evaluation.csv")
    resources = read(RUNS / "logs" / f"{run}.resources.csv")
    utilization = read(RUNS / "logs" / f"{run}.utilization.csv")
    durations = marks_durations(marks)
    start, end = training_window(steps)
    middle = (start + end) / 2
    steady = steps[steps["t"] >= middle]
    steady_samples = samples[samples["t"] >= middle] if not samples.empty else samples
    # The Windows counters over the steady window only (not startup)
    utilization = clipped(utilization, wall_of(marks, middle), wall_of(marks, end))
    resources_steady = clipped(resources, wall_of(marks, middle), wall_of(marks, end))
    total = steps["total_s"].sum()
    steady_total = steady["total_s"].sum()
    test_rounds = {k: v for k, v in durations.items() if "TEST" in k}
    tier_cols = [c for c in samples.columns if c.endswith("_evicted")]
    row = {
        "run": run,
        "build_dojos_min": durations.get("build_dojos", float("nan")) / 60,
        "preflight_min": durations.get("preflight", float("nan")) / 60,
        "build_model_min": durations.get("build_model", float("nan")) / 60,
        "warm_cache_min": durations.get("warm_feature_cache", float("nan")) / 60,
        "train_min": durations.get("train", float("nan")) / 60,
        "steps": len(steps),
        "steps_completed": int(steps["completed"].sum()),
        "cards": int(steps["cards"].sum()),
        "cards_per_s_in_steps": steps["cards"].sum() / total,
        "cards_per_s_steady": steady["cards"].sum() / steady_total,
        "cards_per_s_wall_steady": steady["cards"].sum() / (end - middle),
        "step_median_s": steps["total_s"].median(),
        "step_p90_s": steps["total_s"].quantile(0.9),
        "wait_share": steps["wait_s"].sum() / total,
        "forward_share": steps["forward_s"].sum() / total,
        "remainder_share": steps["remainder_s"].sum() / total,
        "wait_share_median_step": (steps["wait_s"] / steps["total_s"]).median(),
        "wait_share_steady": steady["wait_s"].sum() / steady_total,
        "wait_share_median_step_steady": (steady["wait_s"] / steady["total_s"]).median(),
        "forward_share_steady": steady["forward_s"].sum() / steady_total,
        **{f"{s[:-2]}_share_train_thread": steps[s].sum() / total for s in STAGES},
        **{f"{s[:-2]}_share_steady": steady[s].sum() / steady_total for s in STAGES},
        "live_texts_train_thread": int(steps["live_texts"].sum()),
        "test_pass_mean_s": (sum(test_rounds.values()) / len(test_rounds)) if test_rounds else float("nan"),
        "test_passes": len(test_rounds),
        "hit_rate_training": hit_rate(samples),
        "hit_rate_steady": hit_rate(steady_samples),
        "evicted_total": int(samples[tier_cols].iloc[-1].sum()) if tier_cols and not samples.empty else 0,
        "vram_peak_gib": samples["vram_peak_gib"].max() if not samples.empty else float("nan"),
        "vram_reserved_max_gib": samples["vram_reserved_gib"].max() if not samples.empty else float("nan"),
        "python_vram_max_gb": resources["python_vram_gb"].max() if not resources.empty else float("nan"),
        "python_ws_max_gb": resources["python_ws_gb"].max() if not resources.empty else float("nan"),
        "free_ram_min_gb": resources["free_ram_gb"].min() if not resources.empty else float("nan"),
        "python_ws_median_steady_gb": resources_steady["python_ws_gb"].median() if not resources_steady.empty else float("nan"),
        "gpu_compute_median_pct": utilization["gpu_compute_pct"].median() if not utilization.empty else float("nan"),
        "gpu_compute_p90_pct": utilization["gpu_compute_pct"].quantile(0.9) if not utilization.empty else float("nan"),
        "python_cpu_median_pct": utilization["py_cpu_pct"].median() if not utilization.empty else float("nan"),
        "test_dojos_failed": int((evaluation["completed"] == 0).sum()) if not evaluation.empty else 0,
    }
    return row


def hourly(run: str) -> pd.DataFrame:
    profile = RUNS / run / "profile"
    steps = read(profile / "steps.csv")
    samples = read(profile / "samples.csv")
    utilization = read(RUNS / "logs" / f"{run}.utilization.csv")
    if steps.empty:
        return pd.DataFrame()
    start, _ = training_window(steps)
    steps = steps.assign(hour=((steps["t"] - start) // 3600).astype(int))
    result = steps.groupby("hour").agg(
        steps=("total_s", "size"), cards=("cards", "sum"), step_s=("total_s", "sum"),
        wait_s=("wait_s", "sum"), forward_s=("forward_s", "sum"),
        live_encoding_s=("live_encoding_s", "sum"),
    )
    result["cards_per_s"] = result["cards"] / result["step_s"]
    result["wait_share"] = result["wait_s"] / result["step_s"]
    result["live_share"] = result["live_encoding_s"] / result["step_s"]
    if not samples.empty:
        # Each sample's increase over the one before, credited to its hour
        samples = samples.assign(
            asked=samples["cache_texts"].diff(), encoded=samples["cache_encoded"].diff())
        samples = samples[samples["t"] >= start].assign(
            hour=lambda f: ((f["t"] - start) // 3600).astype(int))
        per = samples.groupby("hour").agg(
            asked=("asked", "sum"), encoded=("encoded", "sum"),
            vram_reserved_gib=("vram_reserved_gib", "max"))
        per["hit_rate"] = 1 - per["encoded"] / per["asked"]
        result = result.join(per[["hit_rate", "vram_reserved_gib"]])
    result.insert(0, "run", run)
    return result.reset_index()


def by_dojo(run: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    profile = RUNS / run / "profile"
    steps = read(profile / "steps.csv")
    builds = read(profile / "builds.csv")
    if steps.empty:
        return pd.DataFrame(), pd.DataFrame()
    total = steps["total_s"].sum()
    dojo = steps.groupby("dojo").agg(
        steps=("total_s", "size"), time_s=("total_s", "sum"), step_median_s=("total_s", "median"),
        wait_median_s=("wait_s", "median"), cards_per_step=("cards", "mean"),
        live_texts_per_step=("live_texts", "mean"), cards=("cards", "sum"))
    dojo["time_share"] = dojo["time_s"] / total
    dojo["cards_per_s"] = dojo["cards"] / dojo["time_s"]
    if not builds.empty:
        builds = builds.assign(build_ms_per_card=1000 * builds["build_s"] / builds["cards"].clip(lower=1))
        per = builds.groupby("dojo").agg(
            build_ms_per_card_median=("build_ms_per_card", "median"),
            build_median_s=("build_s", "median"), passes_started=("pass", "max"))
        dojo = dojo.join(per)
    dojo.insert(0, "run", run)
    dojo = dojo.reset_index().sort_values("time_s", ascending=False)
    families = steps.assign(family=steps["dojo"].map(family)).groupby("family").agg(
        steps=("total_s", "size"), time_s=("total_s", "sum"), step_median_s=("total_s", "median"),
        wait_s=("wait_s", "sum"), cards=("cards", "sum"), live_s=("live_encoding_s", "sum"))
    families["time_share"] = families["time_s"] / total
    families["wait_share"] = families["wait_s"] / families["time_s"]
    families["live_share"] = families["live_s"] / families["time_s"]
    families["cards_per_s"] = families["cards"] / families["time_s"]
    families.insert(0, "run", run)
    return dojo, families.reset_index().sort_values("time_s", ascending=False)


def test_by_dojo(run: str) -> pd.DataFrame:
    evaluation = read(RUNS / run / "profile" / "evaluation.csv")
    if evaluation.empty:
        return evaluation
    result = evaluation.groupby("dojo").agg(
        rounds=("total_s", "size"), mean_s=("total_s", "mean"), cards=("cards", "mean"),
        live_s=("live_encoding_s", "mean"))
    result.insert(0, "run", run)
    return result.reset_index().sort_values("mean_s", ascending=False)


def mtg_projection(summary: pd.DataFrame, card_factor: float) -> pd.DataFrame:
    """Anchor visits needed: every TRAIN-visible card of every TRAIN deck
    once (card_factor: the share of a TRAIN deck's rows that are
    TRAIN-visible known cards; 1.0 gives the upper bound). One step
    anchors every item card in it, so hours = visits / the steady
    wall-clock anchor rate. Decks per card is measured from builds.csv."""
    rows = []
    for _, row in summary[summary["run"].str.startswith("mtg_")].iterrows():
        builds = read(RUNS / row["run"] / "profile" / "builds.csv")
        decks_per_card = builds["examples"].sum() / builds["cards"].sum()
        rate = row["cards_per_s_wall_steady"]
        for factor, label in ((card_factor, "estimate"), (1.0, "upper_bound")):
            visits = MTG_DECK_CARD_ROWS * MTG_TRAIN_SHARE * factor
            rows.append({
                "run": row["run"],
                "visits": label,
                "card_factor": factor,
                "anchor_cards_per_s_wall_steady": rate,
                "anchor_cards_per_s_in_steps_steady": row["cards_per_s_steady"],
                "decks_per_card": decks_per_card,
                "decks_per_s_wall_steady": rate * decks_per_card,
                "anchor_visits_needed": visits,
                "hours_one_anchor_pass": visits / rate / 3600,
                "days_one_anchor_pass": visits / rate / 86400,
                "hours_one_deck_pass": MTG_DECKS * MTG_TRAIN_SHARE / (rate * decks_per_card) / 3600,
            })
    return pd.DataFrame(rows)


def plot(summary: pd.DataFrame, hours: pd.DataFrame) -> None:
    parts = summary.set_index("run")[["wait_share", "forward_share", "remainder_share"]]
    ax = parts.plot.barh(stacked=True, figsize=(7, 0.6 * len(parts) + 1.5))
    ax.set_xlabel("share of summed training step time")
    ax.set_title("Where a step's time goes")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
    plt.tight_layout()
    plt.savefig(RESULTS / "step_parts.png", dpi=120)
    plt.close()
    if hours.empty:
        return
    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    for run, group in hours.groupby("run"):
        axes[0].plot(group["hour"], group["cards_per_s"], marker="o", label=run)
        if "hit_rate" in group:
            axes[1].plot(group["hour"], group["hit_rate"], marker="o", label=run)
    axes[0].set_ylabel("cards/s (in steps)")
    axes[1].set_ylabel("cache hit rate")
    axes[1].set_xlabel("hour of training")
    axes[0].legend()
    plt.tight_layout()
    plt.savefig(RESULTS / "hourly_throughput.png", dpi=120)
    plt.close()


def plot_timeline(runs: list[str], bin_s: float = 300) -> None:
    """Per run, over minutes of training (5-minute bins): wall-clock
    cards/s, the wait share of step time, and texts encoded so far (the
    cache's misses, warm-up excluded)."""
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    for run in runs:
        profile = RUNS / run / "profile"
        steps = read(profile / "steps.csv")
        if steps.empty:
            continue
        start, _ = training_window(steps)
        bins = steps.assign(minute=((steps["t"] - start) // bin_s) * bin_s / 60).groupby("minute").agg(
            cards=("cards", "sum"), step_s=("total_s", "sum"), wait_s=("wait_s", "sum"))
        axes[0].plot(bins.index, bins["cards"] / bin_s, label=run)
        axes[1].plot(bins.index, bins["wait_s"] / bins["step_s"], label=run)
        samples = read(profile / "samples.csv")
        if not samples.empty:
            samples = samples[samples["t"] >= start]
            if not samples.empty:
                axes[2].plot((samples["t"] - start) / 60,
                             samples["cache_encoded"] - samples["cache_encoded"].iloc[0], label=run)
    axes[0].set_ylabel("cards/s (wall clock)")
    axes[1].set_ylabel("wait share of step time")
    axes[2].set_ylabel("texts encoded since training start")
    axes[2].set_xlabel("minutes of training (TEST passes included)")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Throughput, waiting and cache misses over a run")
    plt.tight_layout()
    plt.savefig(RESULTS / "timeline.png", dpi=120)
    plt.close()


def main() -> None:
    runs = sys.argv[1:] or DEFAULT_RUNS
    RESULTS.mkdir(parents=True, exist_ok=True)
    summaries = [row for row in (summarize(run) for run in runs) if row]
    summary = pd.DataFrame(summaries)
    hours = pd.concat([hourly(run) for run in runs], ignore_index=True)
    dojos, families = zip(*(by_dojo(run) for run in runs))
    summary.to_csv(RESULTS / "run_summary.csv", index=False, float_format="%.4g")
    hours.to_csv(RESULTS / "hourly.csv", index=False, float_format="%.4g")
    pd.concat(dojos).to_csv(RESULTS / "by_dojo.csv", index=False, float_format="%.4g")
    pd.concat(families).to_csv(RESULTS / "by_family.csv", index=False, float_format="%.4g")
    pd.concat([test_by_dojo(run) for run in runs]).to_csv(
        RESULTS / "test_by_dojo.csv", index=False, float_format="%.4g")
    card_factor = float(MTG_CARD_FACTOR_FILE.read_text().split()[0]) if MTG_CARD_FACTOR_FILE.exists() else 0.8
    mtg_projection(summary, card_factor).to_csv(RESULTS / "mtg_projection.csv", index=False, float_format="%.4g")
    plot(summary, hours)
    plot_timeline(runs)
    print(summary.T.to_string())


if __name__ == "__main__":
    main()
