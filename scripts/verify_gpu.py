"""Checks whether this machine's GPU can train with PyTorch.

Answers, in order: is a GPU visible, do matmuls run on it and agree with
the CPU, do fp16/bf16 autocast work, does scaled_dot_product_attention run,
and does a small transformer train a few steps. Each check is independent
so one failure does not hide the rest. Ends with throughput and memory
numbers to size a real run against.

Usage (from the project root, in the venv that has the GPU build of torch):

    python scripts/verify_gpu.py
"""

import math
import sys
import time
import traceback
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

Check = Callable[[torch.device], str]


def describe_environment() -> torch.device | None:
    """Print torch/GPU details; return the GPU device, or None if absent."""
    print(f"python {sys.version.split()[0]}  torch {torch.__version__}")
    print(f"hip={getattr(torch.version, 'hip', None)}  cuda={torch.version.cuda}")
    if not torch.cuda.is_available():
        print("NO GPU VISIBLE to torch (torch.cuda.is_available() is False)")
        return None
    props = torch.cuda.get_device_properties(0)
    print(f"device: {props.name}  arch={getattr(props, 'gcnArchName', '?')}")
    print(f"vram: {props.total_memory / 2**30:.1f} GiB")
    return torch.device("cuda:0")


def check_matmul_matches_cpu(device: torch.device) -> str:
    generator = torch.Generator().manual_seed(0)
    a = torch.randn(1024, 1024, generator=generator)
    b = torch.randn(1024, 1024, generator=generator)
    expected = a @ b
    actual = (a.to(device) @ b.to(device)).cpu()
    error = (expected - actual).abs().max().item()
    if error > 1e-2:
        raise AssertionError(f"GPU/CPU matmul disagree, max abs error {error}")
    return f"max abs error vs CPU {error:.2e}"


def check_autocast(dtype: torch.dtype) -> Check:
    def run(device: torch.device) -> str:
        a = torch.randn(2048, 2048, device=device)
        b = torch.randn(2048, 2048, device=device)
        with torch.autocast("cuda", dtype=dtype):
            out = a @ b
        torch.cuda.synchronize()
        if not torch.isfinite(out).all():
            raise AssertionError("non-finite output")
        return f"out dtype {out.dtype}"

    return run


def check_sdpa(device: torch.device) -> str:
    q = torch.randn(4, 8, 512, 64, device=device, dtype=torch.float16)
    out = F.scaled_dot_product_attention(q, q, q)
    torch.cuda.synchronize()
    if not torch.isfinite(out).all():
        raise AssertionError("non-finite output")
    return "fp16 attention ok"


def time_matmul(device: torch.device, dtype: torch.dtype) -> float:
    """Achieved TFLOPS of a square matmul in the given dtype."""
    size = 4096
    a = torch.randn(size, size, device=device, dtype=dtype)
    b = torch.randn(size, size, device=device, dtype=dtype)
    for _ in range(3):
        a @ b
    torch.cuda.synchronize()
    start = time.perf_counter()
    iterations = 20
    for _ in range(iterations):
        a @ b
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return 2 * size**3 * iterations / elapsed / 1e12


def check_throughput(device: torch.device) -> str:
    parts = []
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        try:
            parts.append(f"{dtype}: {time_matmul(device, dtype):.1f} TFLOPS")
        except Exception as error:
            parts.append(f"{dtype}: FAILED ({type(error).__name__})")
    return "; ".join(parts)


def check_training_steps(device: torch.device) -> str:
    """Train a small encoder for a few steps in fp32, then in fp16 autocast."""
    layer = nn.TransformerEncoderLayer(
        d_model=768, nhead=12, dim_feedforward=3072, batch_first=True
    )
    model = nn.TransformerEncoder(layer, num_layers=6).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    batch, length = 16, 512
    results = []
    for label, dtype in (("fp32", None), ("fp16", torch.float16)):
        torch.cuda.reset_peak_memory_stats()
        losses = []
        start = time.perf_counter()
        for _ in range(5):
            x = torch.randn(batch, length, 768, device=device)
            optimizer.zero_grad()
            with torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
                loss = model(x).float().pow(2).mean()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        if not all(math.isfinite(loss_value) for loss_value in losses):
            raise AssertionError(f"{label}: non-finite loss {losses}")
        peak = torch.cuda.max_memory_allocated() / 2**30
        results.append(
            f"{label}: {5 * batch * length / elapsed:,.0f} tok/s, peak {peak:.1f} GiB"
        )
    return "; ".join(results) + f" (6-layer d768 encoder, {batch}x{length})"


def run_check(name: str, check: Check, device: torch.device) -> bool:
    try:
        print(f"[ OK ] {name}: {check(device)}")
        return True
    except Exception:
        print(f"[FAIL] {name}")
        traceback.print_exc(limit=2)
        return False


def main() -> int:
    device = describe_environment()
    if device is None:
        return 1
    checks: list[tuple[str, Check]] = [
        ("matmul matches CPU", check_matmul_matches_cpu),
        ("fp16 autocast", check_autocast(torch.float16)),
        ("bf16 autocast", check_autocast(torch.bfloat16)),
        ("scaled_dot_product_attention", check_sdpa),
        ("matmul throughput", check_throughput),
        ("transformer training steps", check_training_steps),
    ]
    results = [run_check(name, check, device) for name, check in checks]
    print(f"\n{sum(results)}/{len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
