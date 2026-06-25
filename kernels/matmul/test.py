#!/usr/bin/env python3
"""Correctness tests for naive CUDA matmul."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BIN = ROOT / "build" / "naive_matmul"


def run_matmul(
    binary: Path,
    a: np.ndarray,
    b: np.ndarray,
    tmp_dir: Path,
) -> np.ndarray:
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("A and B must be 2-D arrays")

    m, k = a.shape
    k_b, n = b.shape
    if k != k_b:
        raise ValueError(f"Incompatible shapes: A is {a.shape}, B is {b.shape}")

    a_path = tmp_dir / "a.bin"
    b_path = tmp_dir / "b.bin"
    c_path = tmp_dir / "c.bin"

    a.astype(np.float32).tofile(a_path)
    b.astype(np.float32).tofile(b_path)

    cmd = [str(binary), str(m), str(k), str(n), str(a_path), str(b_path), str(c_path)]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "naive_matmul failed\n"
            f"command: {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    return np.fromfile(c_path, dtype=np.float32).reshape(m, n)


def check_case(
    binary: Path,
    m: int,
    k: int,
    n: int,
    tmp_dir: Path,
    seed: int,
    atol: float = 1e-3,
) -> None:
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((m, k), dtype=np.float32)
    b = rng.standard_normal((k, n), dtype=np.float32)

    c_cuda = run_matmul(binary, a, b, tmp_dir)
    c_ref = torch.matmul(torch.from_numpy(a), torch.from_numpy(b)).numpy()

    max_err = np.max(np.abs(c_cuda - c_ref))
    print(f"shape=({m}, {k}) x ({k}, {n})  max_abs_error={max_err:.6e}")

    if max_err > atol:
        raise AssertionError(f"max_abs_error {max_err} exceeds atol {atol}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--binary",
        type=Path,
        default=DEFAULT_BIN,
        help=f"path to naive_matmul binary (default: {DEFAULT_BIN})",
    )
    parser.add_argument(
        "--tmp-dir",
        type=Path,
        default=ROOT / "build" / "matmul_tmp",
        help="directory for temporary matrix files",
    )
    args = parser.parse_args()

    if not args.binary.exists():
        print(f"Binary not found: {args.binary}", file=sys.stderr)
        print("Build first with: cmake -B build && cmake --build build", file=sys.stderr)
        return 1

    args.tmp_dir.mkdir(parents=True, exist_ok=True)

    cases = [
        (64, 64, 64, 0),
        (128, 128, 128, 1),
        (256, 128, 64, 2),
        (77, 33, 91, 3),  # non-multiple-of-16 sizes
    ]

    for m, k, n, seed in cases:
        check_case(args.binary, m, k, n, args.tmp_dir, seed)

    print("All tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
