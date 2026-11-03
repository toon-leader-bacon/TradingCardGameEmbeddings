"""Summarize py-spy raw (collapsed) stacks -> results/pyspy_<run>.csv.

Usage (from project root):
    python experiments/2026-10-29_throughput_shakeout/analysis/pyspy_report.py pyspy_all pyspy_gil

Each raw line is "process ...;process ...;thread (id): name;frame;...;frame count",
leaf last. For each thread: total samples, then the inclusive samples of
every function (a function counts once per stack even if recursive) and
of the leaf (self) function. Writes one CSV per run with columns
run, thread, kind (inclusive|self), function, samples, share_of_thread,
keeping the top 40 per thread and kind; prints the top 15.
"""

import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

_RAW = Path("data/runs/2026-10-29_throughput_shakeout/pyspy")
_RESULTS = Path("experiments/2026-10-29_throughput_shakeout/results")
_THREAD = re.compile(r"^thread \(\w+\)(?:: (.*))?$")
_TOP = 40


def _parse(path: Path) -> tuple[Counter[str], dict[str, Counter[str]], dict[str, Counter[str]]]:
    """Thread totals, inclusive and self counters per thread."""
    totals: Counter[str] = Counter()
    inclusive: dict[str, Counter[str]] = defaultdict(Counter)
    leaf: dict[str, Counter[str]] = defaultdict(Counter)
    for line in open(path, encoding="utf-8", errors="replace"):
        stack, _, count_text = line.rstrip("\n").rpartition(" ")
        if not stack:
            continue
        count = int(count_text)
        frames = stack.split(";")
        thread = "?"
        body: list[str] = []
        for index, frame in enumerate(frames):
            match = _THREAD.match(frame)
            if match:
                thread = match.group(1) or frame
                body = frames[index + 1 :]
                break
        totals[thread] += count
        for function in set(body):
            inclusive[thread][function] += count
        if body:
            leaf[thread][body[-1]] += count
    return totals, inclusive, leaf


def report(run: str) -> None:
    totals, inclusive, leaf = _parse(_RAW / f"{run}.txt")
    rows = []
    print(f"== {run}: {sum(totals.values())} samples")
    for thread, total in totals.most_common():
        print(f"-- thread {thread}: {total} samples")
        for kind, counter in (("inclusive", inclusive[thread]), ("self", leaf[thread])):
            for rank, (function, samples) in enumerate(counter.most_common(_TOP)):
                rows.append(
                    {"run": run, "thread": thread, "kind": kind, "function": function,
                     "samples": samples, "share_of_thread": round(samples / total, 4)}
                )
                if rank < 15 and total > 200:
                    print(f"  {kind:9s} {samples / total:6.1%} {function[:150]}")
    _RESULTS.mkdir(parents=True, exist_ok=True)
    with open(_RESULTS / f"pyspy_{run}.csv", "w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    for name in sys.argv[1:]:
        report(name)
