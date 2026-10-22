"""Run scripts/run_training.py with timers wrapped around the training
system's hot paths, writing per-step, per-batch-build and per-TEST-dojo
timing CSVs. The source tree is not changed: the timers are installed by
replacing a few attributes before main() runs.

Usage (from the project root, ROCm venv):

    PYTHONPATH=. python experiments/2026-10-21_cache_shakeout/analysis/profiled_training.py \\
        --profile-out data/runs/2026-10-21_cache_shakeout/profile/<run> \\
        [--no-binder-copy] -- <config> [run_training.py arguments]

Writes into --profile-out:
    steps.csv       one row per training step (main thread): dojo, cards,
                    wait for the batch, forward + loss, backward + optimizer,
                    live-encoded texts and their time, cache lookup time,
                    serialization time
    builds.csv      one row per TRAIN batch built (prefetch worker, or the
                    main thread without prefetch): dojo, cards, build time,
                    of which mods and CardBinder deep copies; reserve time
                    and the texts it encoded
    test.csv        one row per (round, dojo) TEST pass: time, examples,
                    cards, live-encoded texts, build-side time
    phases.csv      startup and per-round wall-clock marks
    memory.csv      torch's allocated / reserved / peak VRAM every 10 s
    binder_check.txt  with --no-binder-copy: whether every binder card
                    still serializes as it did before training

--no-binder-copy makes CardBinder return its stored cards instead of deep
copies, and checks at the end that training mutated none of them.

Timers synchronize the current CUDA stream before reading the clock, so
GPU work is charged to the stage that queued it. The training thread
already synchronizes once per step (the gradient norm's .item()), so this
adds little there; the prefetch worker's syncs block only the worker.
"""

import argparse
import copy
import csv
import hashlib
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

import run_training  # noqa: E402  (scripts/run_training.py)
from src.data_refinement.card_binder import card_binder as card_binder_module  # noqa: E402
from src.dojos.mods.mod_pipeline import ModPipeline  # noqa: E402
from src.encoder_model import card_embedding_pipeline as pipeline_module  # noqa: E402
from src.encoder_model.card_embedding_pipeline import CardEmbeddingPipeline  # noqa: E402
from src.encoder_model.feature_cache.pooled_feature_cache import PooledFeatureCache  # noqa: E402
from src.schema.type_hints import iter_cards  # noqa: E402
from src.training import round_evaluation  # noqa: E402
from src.training import trainer as trainer_module  # noqa: E402
from src.training.diet import prefetching_batch_streams as prefetch_module  # noqa: E402
from src.training.diet.dojo_batch_stream import DojoBatchStream  # noqa: E402

_COUNTERS = ("deepcopy_s", "deepcopies", "mods_s", "serialize_s", "lookup_s",
             "live_s", "live_texts")


class ThreadCounters(threading.local):
    """Per-thread running totals; a record takes deltas around its work."""

    def __init__(self) -> None:
        self.totals: dict[str, float] = defaultdict(float)
        self.mod_depth = 0
        self.lookup_depth = 0

    def snapshot(self) -> dict[str, float]:
        return {name: self.totals[name] for name in _COUNTERS}


COUNTERS = ThreadCounters()


def deltas(before: dict[str, float]) -> dict[str, float]:
    after = COUNTERS.snapshot()
    return {name: after[name] - before[name] for name in _COUNTERS}


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.current_stream().synchronize()


class CsvLog:
    """A CSV written row by row, flushed every row (a crash keeps the rows)."""

    def __init__(self, path: Path, fields: list[str]) -> None:
        self._handle = path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._handle, fieldnames=fields)
        self._writer.writeheader()
        self._lock = threading.Lock()

    def write(self, row: dict[str, Any]) -> None:
        with self._lock:
            self._writer.writerow(
                {k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in row.items()}
            )
            self._handle.flush()

    def close(self) -> None:
        self._handle.close()


class Profiler:
    def __init__(self, out: Path, binder_copy: bool) -> None:
        out.mkdir(parents=True, exist_ok=True)
        self.out = out
        self.binder_copy = binder_copy
        self.started = time.monotonic()
        self.round_index = 0
        self.step_index = 0
        self.steps = CsvLog(out / "steps.csv", [
            "t", "round", "step", "dojo", "cards", "ran", "total_s", "wait_s",
            "forward_loss_s", "backward_opt_s", *_COUNTERS])
        self.builds = CsvLog(out / "builds.csv", [
            "t", "thread", "dojo", "examples", "cards", "pass_id", "build_s", "reserve_s",
            "reserve_live_texts", "reserve_live_s", *_COUNTERS])
        self.test = CsvLog(out / "test.csv", [
            "t", "round", "dojo", "examples", "cards", "total_s", *_COUNTERS])
        self.phases = CsvLog(out / "phases.csv", ["t", "wall", "mark"])
        self.memory = CsvLog(out / "memory.csv", [
            "t", "allocated_gib", "reserved_gib", "peak_allocated_gib"])
        self._stop = threading.Event()
        self.mark("profiler start")

    def now(self) -> float:
        return time.monotonic() - self.started

    def mark(self, what: str) -> None:
        self.phases.write({"t": self.now(), "wall": time.strftime("%Y-%m-%dT%H:%M:%S"),
                           "mark": what})

    def sample_memory(self) -> None:
        while not self._stop.wait(10.0):
            if not torch.cuda.is_available():
                continue
            gib = 1024**3
            self.memory.write({
                "t": self.now(),
                "allocated_gib": torch.cuda.memory_allocated() / gib,
                "reserved_gib": torch.cuda.memory_reserved() / gib,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / gib,
            })

    def close(self) -> None:
        self._stop.set()
        self.mark("profiler end")
        for log in (self.steps, self.builds, self.test, self.phases, self.memory):
            log.close()


def card_count(inputs: Any) -> int:
    return sum(1 for _ in iter_cards(inputs))


def install(profiler: Profiler) -> None:
    """Replace the hot paths with timed wrappers."""

    # CardBinder's deep copies: timed, or skipped with --no-binder-copy
    real_deepcopy = copy.deepcopy

    def binder_deepcopy(value: Any, memo: Any = None) -> Any:
        start = time.perf_counter()
        result = real_deepcopy(value, memo) if profiler.binder_copy else value
        COUNTERS.totals["deepcopy_s"] += time.perf_counter() - start
        COUNTERS.totals["deepcopies"] += 1
        return result

    card_binder_module.copy = SimpleNamespace(deepcopy=binder_deepcopy)  # type: ignore[attr-defined]

    # Mods: the outermost ModPipeline call per thread
    def timed_mods(real: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            COUNTERS.mod_depth += 1
            start = time.perf_counter()
            try:
                return real(self, *args, **kwargs)
            finally:
                COUNTERS.mod_depth -= 1
                if COUNTERS.mod_depth == 0:
                    COUNTERS.totals["mods_s"] += time.perf_counter() - start
        return wrapper

    ModPipeline.apply = timed_mods(ModPipeline.apply)  # type: ignore[method-assign]
    ModPipeline.apply_single = timed_mods(ModPipeline.apply_single)  # type: ignore[method-assign]

    # Serialization of each card before the cache lookup
    real_serialize = pipeline_module.serialize_card_to_json

    def timed_serialize(card: Any) -> str:
        start = time.perf_counter()
        try:
            return real_serialize(card)
        finally:
            COUNTERS.totals["serialize_s"] += time.perf_counter() - start

    pipeline_module.serialize_card_to_json = timed_serialize  # type: ignore[assignment]

    # The frozen stages on texts (a cache miss, or every card uncached)
    real_pool_live = CardEmbeddingPipeline._pool_live

    def timed_pool_live(self: Any, texts: list[str]) -> torch.Tensor:
        sync()
        start = time.perf_counter()
        result = real_pool_live(self, texts)
        sync()
        COUNTERS.totals["live_s"] += time.perf_counter() - start
        COUNTERS.totals["live_texts"] += len(texts)
        return result

    CardEmbeddingPipeline._pool_live = timed_pool_live  # type: ignore[method-assign]

    # Cache lookups (features and reserve), live encoding included
    def timed_lookup(real: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            COUNTERS.lookup_depth += 1
            start = time.perf_counter()
            try:
                result = real(self, *args, **kwargs)
                sync()
                return result
            finally:
                COUNTERS.lookup_depth -= 1
                if COUNTERS.lookup_depth == 0:
                    COUNTERS.totals["lookup_s"] += time.perf_counter() - start
        return wrapper

    PooledFeatureCache.features = timed_lookup(PooledFeatureCache.features)  # type: ignore[method-assign]
    PooledFeatureCache.reserve = timed_lookup(PooledFeatureCache.reserve)  # type: ignore[method-assign]

    # Batch building, on whichever thread builds
    real_next_batch = DojoBatchStream.next_batch

    def timed_build(self: DojoBatchStream) -> Any:
        before = COUNTERS.snapshot()
        start = time.perf_counter()
        batch = real_next_batch(self)
        elapsed = time.perf_counter() - start
        profiler.builds.write({
            "t": profiler.now(), "thread": threading.current_thread().name,
            "dojo": self._dojo.name, "examples": len(batch),
            "cards": card_count(batch.inputs), "pass_id": id(self._batches),
            "build_s": elapsed, "reserve_s": "", "reserve_live_texts": "",
            "reserve_live_s": "", **deltas(before)})
        return batch

    DojoBatchStream.next_batch = timed_build  # type: ignore[method-assign]

    # Reservation (pin + warm) of each built batch, on the worker
    real_reserve = prefetch_module.PrefetchingBatchStreams._reserve_cards

    def timed_reserve(self: Any, name: str, batch: Any) -> Any:
        before = COUNTERS.snapshot()
        start = time.perf_counter()
        result = real_reserve(self, name, batch)
        moved = deltas(before)
        profiler.builds.write({
            "t": profiler.now(), "thread": threading.current_thread().name,
            "dojo": name, "examples": "", "cards": "", "pass_id": "", "build_s": "",
            "reserve_s": time.perf_counter() - start,
            "reserve_live_texts": moved["live_texts"],
            "reserve_live_s": moved["live_s"],
            **{k: "" for k in _COUNTERS}})
        return result

    prefetch_module.PrefetchingBatchStreams._reserve_cards = timed_reserve  # type: ignore[method-assign]

    # One training step: wait, forward + loss, the rest
    real_take_step = trainer_module.Trainer._take_step
    real_loss_of = trainer_module.Trainer._loss_of
    step_state = threading.local()

    def timed_loss_of(self: Any, dojo: Any, batch: Any) -> Any:
        step_state.cards = card_count(batch.inputs)
        step_state.wait_s = time.perf_counter() - step_state.start
        start = time.perf_counter()
        result = real_loss_of(self, dojo, batch)
        sync()
        step_state.forward_s = time.perf_counter() - start
        return result

    def timed_take_step(self: Any, phase_run: Any, dojo: Any) -> bool:
        before = COUNTERS.snapshot()
        step_state.start = time.perf_counter()
        step_state.cards = 0
        step_state.wait_s = float("nan")
        step_state.forward_s = float("nan")
        ran = False
        try:
            ran = real_take_step(self, phase_run, dojo)
            sync()
            return ran
        finally:
            total = time.perf_counter() - step_state.start
            wait_s = step_state.wait_s
            if wait_s != wait_s:  # skipped before the forward: all of it was the wait
                wait_s = total
            backward = total - wait_s - (step_state.forward_s if step_state.forward_s == step_state.forward_s else 0.0)
            profiler.steps.write({
                "t": profiler.now(), "round": profiler.round_index,
                "step": profiler.step_index, "dojo": dojo.name,
                "cards": step_state.cards, "ran": int(bool(ran)), "total_s": total,
                "wait_s": wait_s, "forward_loss_s": step_state.forward_s,
                "backward_opt_s": backward, **deltas(before)})
            profiler.step_index += 1

    trainer_module.Trainer._take_step = timed_take_step  # type: ignore[method-assign]
    trainer_module.Trainer._loss_of = timed_loss_of  # type: ignore[method-assign]

    # Each round's TEST pass, per dojo
    real_split_loss = round_evaluation._split_loss
    real_evaluate_round = trainer_module.Trainer._evaluate_round

    def timed_split_loss(model: Any, dojo: Any, split: Any, budget: Any, max_examples: int) -> Any:
        before = COUNTERS.snapshot()
        start = time.perf_counter()
        examples = 0
        cards = 0
        real_batches = dojo.batches

        def counting_batches(*args: Any, **kwargs: Any) -> Any:
            nonlocal examples, cards
            for batch in real_batches(*args, **kwargs):
                examples += len(batch)
                cards += card_count(batch.inputs)
                yield batch

        dojo.batches = counting_batches
        try:
            return real_split_loss(model, dojo, split, budget, max_examples)
        finally:
            del dojo.batches
            sync()
            profiler.test.write({
                "t": profiler.now(), "round": profiler.round_index, "dojo": dojo.name,
                "examples": examples, "cards": cards,
                "total_s": time.perf_counter() - start, **deltas(before)})

    def timed_evaluate_round(self: Any, phase_run: Any, round_index: int, run_started: float) -> Any:
        profiler.mark(f"round {round_index} training end; TEST start")
        try:
            return real_evaluate_round(self, phase_run, round_index, run_started)
        finally:
            profiler.mark(f"round {round_index} TEST end")
            profiler.round_index = round_index + 1

    round_evaluation._split_loss = timed_split_loss  # type: ignore[assignment]
    trainer_module.Trainer._evaluate_round = timed_evaluate_round  # type: ignore[method-assign]

    # Startup stages of scripts/run_training.main
    for name in ("build_dojos", "preflight_passes", "build_model", "warm_feature_cache", "train"):
        real = getattr(run_training, name)

        def stage(*args: Any, _real: Callable[..., Any] = real, _name: str = name, **kwargs: Any) -> Any:
            profiler.mark(f"{_name} start")
            try:
                return _real(*args, **kwargs)
            finally:
                profiler.mark(f"{_name} end")

        setattr(run_training, name, stage)


def binder_fingerprints() -> dict[str, str]:
    """Each loaded binder's cards, hashed card by card from their serialized
    text (in the binder's own storage, never a copy)."""
    result: dict[str, str] = {}
    for binder in BINDERS:
        digest = hashlib.blake2b(digest_size=16)
        for uuid, card in sorted(binder._cards_by_uuid.items(), key=lambda kv: str(kv[0])):
            digest.update(str(uuid).encode())
            digest.update(REAL_SERIALIZE(card).encode())
        result[f"{id(binder)}"] = digest.hexdigest()
    return result


BINDERS: list[Any] = []
REAL_SERIALIZE = pipeline_module.serialize_card_to_json


def track_binders() -> None:
    real_init = card_binder_module.CardBinder.__init__

    def init(self: Any, *args: Any, **kwargs: Any) -> None:
        real_init(self, *args, **kwargs)
        BINDERS.append(self)

    card_binder_module.CardBinder.__init__ = init  # type: ignore[method-assign]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-out", type=Path, required=True)
    parser.add_argument("--no-binder-copy", action="store_true")
    args, rest = parser.parse_known_args()
    if rest and rest[0] == "--":
        rest = rest[1:]
    profiler = Profiler(args.profile_out, binder_copy=not args.no_binder_copy)
    track_binders()
    install(profiler)
    sampler = threading.Thread(target=profiler.sample_memory, name="memory-sampler", daemon=True)
    sampler.start()

    # With --no-binder-copy, fingerprint every binder before training starts
    before: dict[str, str] = {}
    if args.no_binder_copy:
        real_train = run_training.train

        def train_with_check(*a: Any, **k: Any) -> Any:
            before.update(binder_fingerprints())
            return real_train(*a, **k)

        run_training.train = train_with_check

    sys.argv = ["run_training.py", *rest]
    try:
        return run_training.main()
    finally:
        if args.no_binder_copy:
            after = binder_fingerprints()
            changed = [key for key in before if before[key] != after.get(key)]
            (args.profile_out / "binder_check.txt").write_text(
                f"binders checked: {len(before)}\nbinders changed: {len(changed)}\n"
                f"result: {'MUTATED' if changed else 'unchanged'}\n",
                encoding="utf-8")
        profiler.close()


if __name__ == "__main__":
    raise SystemExit(main())
