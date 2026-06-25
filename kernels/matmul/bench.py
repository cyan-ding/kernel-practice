#!/usr/bin/env python3
"""Benchmark naive CUDA matmul against PyTorch."""

from __future__ import annotations

import argparse
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BIN = ROOT / "build" / "naive_matmul"


def bench_cuda(
    binary: Path,
    a: np.ndarray,
    b: np.ndarray,
    tmp_dir: Path,
    warmup: int,
    iters: int,
) -> float:
    m, k = a.shape
    _, n = b.shape

    a_path = tmp_dir / "a.bin"
    b_path = tmp_dir / "b.bin"
    c_path = tmp_dir / "c.bin"
    a.astype(np.float32).tofile(a_path)
    b.astype(np.float32).tofile(b_path)

    cmd = [str(binary), str(m), str(k), str(n), str(a_path), str(b_path), str(c_path)]

    for _ in range(warmup):
        subprocess.run(cmd, check=True, capture_output=True)

    times = []
    for _ in range(iters):
        start = time.perf_counter()
        subprocess.run(cmd, check=True, capture_output=True)
        times.append(time.perf_counter() - start)

    return statistics.median(times)


def bench_torch(a: torch.Tensor, b: torch.Tensor, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        torch.matmul(a, b)
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        start = time.perf_counter()
        torch.matmul(a, b)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    return statistics.median(times)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BIN)
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument(
        "--tmp-dir",
        type=Path,
        default=ROOT / "build" / "matmul_tmp",
    )
    args = parser.parse_args()

    if not args.binary.exists():
        print(f"Binary not found: {args.binary}", file=sys.stderr)
        return 1

    if not torch.cuda.is_available():
        print("CUDA is not available to PyTorch.", file=sys.stderr)
        return 1

    args.tmp_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    a_np = rng.standard_normal((args.m, args.k), dtype=np.float32)
    b_np = rng.standard_normal((args.k, args.n), dtype=np.float32)

    a_torch = torch.from_numpy(a_np).cuda()
    b_torch = torch.from_numpy(b_np).cuda()

    cuda_s = bench_cuda(args.binary, a_np, b_np, args.tmp_dir, args.warmup, args.iters)
    torch_s = bench_torch(a_torch, b_torch, args.warmup, args.iters)

    flops = 2.0 * args.m * args.k * args.n
    print(f"Problem size: ({args.m}, {args.k}) x ({args.k}, {args.n})")
    print(f"naive_matmul: {cuda_s * 1e3:.2f} ms  ({flops / cuda_s / 1e9:.2f} GFLOPS)")
    print(f"torch.matmul: {torch_s * 1e3:.2f} ms  ({flops / torch_s / 1e9:.2f} GFLOPS)")
    print(f"torch speedup: {cuda_s / torch_s:.2f}x")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
