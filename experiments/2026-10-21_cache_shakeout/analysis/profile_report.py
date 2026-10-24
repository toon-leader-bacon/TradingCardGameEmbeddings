"""Summarize the profiler's CSVs for every finished run into results/.

Reads, per run, under data/runs/2026-10-21_cache_shakeout/:
    profile/<run>/{steps,builds,test,phases,memory}.csv   (profiled_training.py)
    <run>/rounds.csv                                      (the run itself)
    logs/<run>.{resources,utilization}.csv                (Windows counters)

Writes into experiments/2026-10-21_cache_shakeout/results/:
    run_summary.csv       one row per run: startup stages, training and TEST
                          time per round, step time and its parts, live-encode
                          share, deep-copy share of building, memory, GPU use
    step_by_family.csv    per run and dojo family: step time and its parts
    build_by_dojo.csv     per run and dojo: TRAIN batch building per card,
                          its mods and deep copies, passes over TRAIN data
    test_by_family.csv    per run, round and dojo family: TEST pass time
    test_by_dojo.csv      per run and dojo: TEST time per round
    loss_pairs.csv        per pair of runs: differences in normalized TEST
                          loss per dojo (median and max absolute)

Usage (from the project root):
    python experiments/2026-10-21_cache_shakeout/analysis/profile_report.py [run ...]
"""

import csv
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Iterable

RUNS = Path("data/runs/2026-10-21_cache_shakeout")
RESULTS = Path("experiments/2026-10-21_cache_shakeout/results")
DEFAULT_RUNS = ["pilot", "cached", "cached_nomods", "cached_nocopy", "uncached"]


def rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def num(value: str | None) -> float:
    if value is None or value == "":
        return float("nan")
    return float(value)


def family(dojo: str) -> str:
    """contrastive dojos by style; metric dojos by source."""
    source = dojo.split(".", 1)[0]
    return source if source.startswith("contrastive") else f"metric:{source}"


def kind(dojo: str) -> str:
    return "contrastive" if dojo.startswith("contrastive") else "metric"


def quantile(values: list[float], q: float) -> float:
    values = sorted(v for v in values if v == v)
    if not values:
        return float("nan")
    index = min(len(values) - 1, max(0, round(q * (len(values) - 1))))
    return values[index]


def fmt(value: float, digits: int = 4) -> str:
    return "" if value != value else f"{value:.{digits}f}"


def write(name: str, fields: list[str], table: Iterable[dict[str, object]]) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in table:
            writer.writerow({k: fmt(v) if isinstance(v, float) else v for k, v in row.items()})


def stage_seconds(phases: list[dict[str, str]]) -> dict[str, float]:
    starts: dict[str, float] = {}
    result: dict[str, float] = {}
    for row in phases:
        mark = row["mark"]
        if mark.endswith(" start"):
            starts[mark[: -len(" start")]] = num(row["t"])
        elif mark.endswith(" end") and mark[: -len(" end")] in starts:
            result[mark[: -len(" end")]] = num(row["t"]) - starts[mark[: -len(" end")]]
    # TEST pass marks: "round N training end; TEST start" .. "round N TEST end"
    test_start: dict[str, float] = {}
    for row in phases:
        mark = row["mark"]
        if "TEST start" in mark:
            test_start[mark.split()[1]] = num(row["t"])
        elif mark.endswith("TEST end"):
            index = mark.split()[1]
            result[f"test_round{index}"] = num(row["t"]) - test_start.get(index, float("nan"))
    return result


def sampler_stats(run: str, start: str | None, end: str | None) -> dict[str, float]:
    """Medians and peaks of the Windows counters over [start, end]."""
    def within(row: dict[str, str]) -> bool:
        return (start is None or row["time"] >= start) and (end is None or row["time"] <= end)

    resources = [r for r in rows(RUNS / "logs" / f"{run}.resources.csv") if within(r)]
    utilization = [r for r in rows(RUNS / "logs" / f"{run}.utilization.csv") if within(r)]
    gpu = [num(r["gpu_compute_pct"]) for r in utilization]
    cpu = [num(r["py_cpu_pct"]) for r in utilization]
    return {
        "peak_py_ram_gb": max((num(r["python_ws_gb"]) for r in resources), default=float("nan")),
        "min_free_ram_gb": min((num(r["free_ram_gb"]) for r in resources), default=float("nan")),
        "peak_py_vram_gb": max((num(r["python_vram_gb"]) for r in resources), default=float("nan")),
        "gpu_compute_median": statistics.median(gpu) if gpu else float("nan"),
        "gpu_compute_p90": quantile(gpu, 0.9),
        "py_cpu_median": statistics.median(cpu) if cpu else float("nan"),
        "n_util_samples": float(len(utilization)),
    }


def wall_of(phases: list[dict[str, str]], mark_part: str, last: bool = False) -> str | None:
    hits = [row["wall"] for row in phases if mark_part in row["mark"]]
    if not hits:
        return None
    return hits[-1] if last else hits[0]


def summarize(run: str) -> dict[str, object] | None:
    profile = RUNS / "profile" / run
    steps = rows(profile / "steps.csv")
    if not steps:
        return None
    builds = [b for b in rows(profile / "builds.csv") if b["build_s"] != ""]
    reserves = [b for b in rows(profile / "builds.csv") if b["reserve_s"] != ""]
    phases = rows(profile / "phases.csv")
    memory = rows(profile / "memory.csv")
    stages = stage_seconds(phases)
    ran = [s for s in steps if s["ran"] == "1"]
    total = sum(num(s["total_s"]) for s in steps)
    cards = sum(num(s["cards"]) for s in ran)
    live_main = sum(num(s["live_s"]) for s in steps)
    live_texts_main = sum(num(s["live_texts"]) for s in steps)
    build_total = sum(num(b["build_s"]) for b in builds)
    build_cards = sum(num(b["cards"]) for b in builds)
    rounds = sorted({s["round"] for s in steps})
    per_round_train = {
        r: sum(num(s["total_s"]) for s in steps if s["round"] == r) for r in rounds
    }
    # Sampler window: training only (from train start to the end)
    window = sampler_stats(run, wall_of(phases, "train start"), wall_of(phases, "train end", last=True))
    summary: dict[str, object] = {
        "run": run,
        "steps": len(steps),
        "steps_ran": len(ran),
        "startup_build_dojos_s": stages.get("build_dojos", float("nan")),
        "startup_preflight_s": stages.get("preflight_passes", float("nan")),
        "startup_build_model_s": stages.get("build_model", float("nan")),
        "startup_warm_cache_s": stages.get("warm_feature_cache", float("nan")),
        "train_total_s": stages.get("train", float("nan")),
        "steps_s_total": total,
        "step_mean_s": total / len(steps),
        "step_median_s": statistics.median(num(s["total_s"]) for s in steps),
        "step_p90_s": quantile([num(s["total_s"]) for s in steps], 0.9),
        "cards_per_s": cards / total if total else float("nan"),
        "wait_share": sum(num(s["wait_s"]) for s in steps) / total,
        "forward_loss_share": sum(num(s["forward_loss_s"]) for s in ran) / total,
        "backward_opt_share": sum(num(s["backward_opt_s"]) for s in ran) / total,
        "live_share_of_step": live_main / total,
        "live_texts_per_step_card": live_texts_main / cards if cards else float("nan"),
        "live_ms_per_text": 1000 * live_main / live_texts_main if live_texts_main else float("nan"),
        "lookup_share_of_step": sum(num(s["lookup_s"]) for s in steps) / total,
        "serialize_share_of_step": sum(num(s["serialize_s"]) for s in steps) / total,
        "build_ms_per_card": 1000 * build_total / build_cards if build_cards else float("nan"),
        "mods_share_of_build": sum(num(b["mods_s"]) for b in builds) / build_total if build_total else float("nan"),
        "deepcopy_share_of_build": sum(num(b["deepcopy_s"]) for b in builds) / build_total if build_total else float("nan"),
        "deepcopy_us_each": 1e6 * sum(num(b["deepcopy_s"]) for b in builds) / max(1.0, sum(num(b["deepcopies"]) for b in builds)),
        "worker_build_s": build_total,
        "worker_reserve_s": sum(num(r["reserve_s"]) for r in reserves),
        "reserve_live_texts": sum(num(r["reserve_live_texts"]) for r in reserves),
        "peak_torch_alloc_gib": max((num(m["peak_allocated_gib"]) for m in memory), default=float("nan")),
        "peak_torch_reserved_gib": max((num(m["reserved_gib"]) for m in memory), default=float("nan")),
        **window,
    }
    for r in rounds:
        summary[f"train_round{r}_s"] = per_round_train[r]
        summary[f"test_round{r}_s"] = stages.get(f"test_round{r}", float("nan"))
    return summary


def step_by_family(run: str) -> list[dict[str, object]]:
    steps = rows(RUNS / "profile" / run / "steps.csv")
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for s in steps:
        groups[family(s["dojo"])].append(s)
    table = []
    for name, group in sorted(groups.items()):
        total = sum(num(s["total_s"]) for s in group)
        cards = sum(num(s["cards"]) for s in group)
        table.append({
            "run": run, "family": name, "steps": len(group),
            "share_of_steps": len(group) / len(steps), "share_of_time": total / sum(num(s["total_s"]) for s in steps),
            "step_median_ms": 1000 * statistics.median(num(s["total_s"]) for s in group),
            "cards_mean": cards / len(group),
            "wait_ms": 1000 * statistics.mean(num(s["wait_s"]) for s in group),
            "forward_loss_ms": 1000 * statistics.mean(num(s["forward_loss_s"]) if s["forward_loss_s"] not in ("", "nan") else 0.0 for s in group),
            "backward_opt_ms": 1000 * statistics.mean(num(s["backward_opt_s"]) for s in group),
            "live_ms": 1000 * statistics.mean(num(s["live_s"]) for s in group),
            "live_texts_per_card": sum(num(s["live_texts"]) for s in group) / cards if cards else float("nan"),
            "lookup_ms": 1000 * statistics.mean(num(s["lookup_s"]) for s in group),
            "serialize_ms": 1000 * statistics.mean(num(s["serialize_s"]) for s in group),
        })
    return table


def build_by_dojo(run: str) -> list[dict[str, object]]:
    builds = [b for b in rows(RUNS / "profile" / run / "builds.csv") if b["build_s"] != ""]
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for b in builds:
        groups[b["dojo"]].append(b)
    table = []
    for dojo, group in sorted(groups.items()):
        cards = sum(num(b["cards"]) for b in group)
        build = sum(num(b["build_s"]) for b in group)
        passes = float("nan")
        if group and "pass_id" in group[0]:
            ids = [b["pass_id"] for b in group]
            passes = 1 + sum(1 for a, b in zip(ids, ids[1:]) if a != b)
        table.append({
            "run": run, "dojo": dojo, "family": family(dojo), "builds": len(group),
            "examples_mean": statistics.mean(num(b.get("examples")) for b in group) if "examples" in group[0] else float("nan"),
            "cards_mean": cards / len(group),
            "build_ms_mean": 1000 * build / len(group),
            "build_ms_per_card": 1000 * build / cards if cards else float("nan"),
            "mods_ms_per_card": 1000 * sum(num(b["mods_s"]) for b in group) / cards if cards else float("nan"),
            "deepcopy_ms_per_card": 1000 * sum(num(b["deepcopy_s"]) for b in group) / cards if cards else float("nan"),
            "deepcopies_per_card": sum(num(b["deepcopies"]) for b in group) / cards if cards else float("nan"),
            "passes_started": passes,
        })
    return table


def test_tables(run: str) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    test = rows(RUNS / "profile" / run / "test.csv")
    by_family: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for t in test:
        by_family[(t["round"], family(t["dojo"]))].append(t)
    families = [{
        "run": run, "round": r, "family": f, "dojos": len(g),
        "seconds": sum(num(t["total_s"]) for t in g),
        "cards": sum(num(t["cards"]) for t in g),
        "ms_per_card": 1000 * sum(num(t["total_s"]) for t in g) / max(1.0, sum(num(t["cards"]) for t in g)),
        "live_share": sum(num(t["live_s"]) for t in g) / max(1e-9, sum(num(t["total_s"]) for t in g)),
        "deepcopy_share": sum(num(t["deepcopy_s"]) for t in g) / max(1e-9, sum(num(t["total_s"]) for t in g)),
    } for (r, f), g in sorted(by_family.items())]
    dojos = [{
        "run": run, "round": t["round"], "dojo": t["dojo"], "seconds": num(t["total_s"]),
        "examples": t["examples"], "cards": t["cards"],
        "live_s": num(t["live_s"]), "deepcopy_s": num(t["deepcopy_s"]), "mods_s": num(t["mods_s"]),
    } for t in test]
    return families, dojos


def last_round_losses(run: str) -> dict[str, float]:
    table = rows(RUNS / run / "rounds.csv")
    if not table:
        return {}
    last = max(int(r["round_index"]) for r in table)
    return {r["dojo"]: num(r["normalized_test_loss"]) for r in table if int(r["round_index"]) == last}


def loss_pairs(runs: list[str]) -> list[dict[str, object]]:
    losses = {run: last_round_losses(run) for run in runs}
    table = []
    for a, b in combinations(runs, 2):
        shared = sorted(set(losses[a]) & set(losses[b]))
        if not shared:
            continue
        for group_name, keep in (("all", lambda d: True), ("contrastive", lambda d: kind(d) == "contrastive"), ("metric", lambda d: kind(d) == "metric")):
            diffs = [abs(losses[a][d] - losses[b][d]) for d in shared if keep(d)]
            signed = [losses[a][d] - losses[b][d] for d in shared if keep(d)]
            if not diffs:
                continue
            table.append({
                "run_a": a, "run_b": b, "dojos": group_name, "n": len(diffs),
                "median_abs_diff": statistics.median(diffs), "p90_abs_diff": quantile(diffs, 0.9),
                "max_abs_diff": max(diffs), "mean_signed_diff_a_minus_b": statistics.mean(signed),
                "median_a": statistics.median(losses[a][d] for d in shared if keep(d)),
                "median_b": statistics.median(losses[b][d] for d in shared if keep(d)),
            })
    return table


def main() -> None:
    runs = sys.argv[1:] or DEFAULT_RUNS
    summaries = [s for s in (summarize(run) for run in runs) if s is not None]
    done = [str(s["run"]) for s in summaries]
    fields: list[str] = []
    for s in summaries:
        fields += [k for k in s if k not in fields]
    write("run_summary.csv", fields, summaries)
    step_rows = [row for run in done for row in step_by_family(run)]
    if step_rows:
        write("step_by_family.csv", list(step_rows[0]), step_rows)
    build_rows = [row for run in done for row in build_by_dojo(run)]
    if build_rows:
        write("build_by_dojo.csv", list(build_rows[0]), build_rows)
    family_rows: list[dict[str, object]] = []
    dojo_rows: list[dict[str, object]] = []
    for run in done:
        f, d = test_tables(run)
        family_rows += f
        dojo_rows += d
    if family_rows:
        write("test_by_family.csv", list(family_rows[0]), family_rows)
        write("test_by_dojo.csv", list(dojo_rows[0]), dojo_rows)
    pairs = loss_pairs([r for r in done if r != "pilot"])
    if pairs:
        write("loss_pairs.csv", list(pairs[0]), pairs)
    print(f"summarized: {', '.join(done)} ({datetime.now():%H:%M})")


if __name__ == "__main__":
    main()
