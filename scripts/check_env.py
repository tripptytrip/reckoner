"""Environment sanity check: which torch is installed, and on which device.

Run after any environment change. The failure this exists to catch is the
expensive one from AGENTS.md §2: a bare ``pip install torch`` pulls CUDA wheels
from PyPI, they import fine on a machine with no NVIDIA hardware, and then
everything runs CPU-only forever while looking healthy. A ``+cu`` suffix on this
box is *always* wrong.

reckoner is CPU-first through chunk 6 (pattern-matching movegen is CPU-heavy and
the model is 2-7M parameters), so a CPU build is the expected, correct result
here — this script reports it as such rather than warning about it. From chunk 7
the ROCm variant becomes worth installing; see pyproject.toml.
"""

from __future__ import annotations

import argparse
import platform
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matmul-n",
        type=int,
        default=2048,
        help="square matmul size for the throughput probe (0 to skip)",
    )
    args = parser.parse_args()

    print(f"python      : {platform.python_version()} ({sys.executable})")

    try:
        import torch
    except ImportError:
        print("torch       : NOT INSTALLED — run `make install`")
        return 1

    print(f"torch       : {torch.__version__}")

    build = "cpu"
    if "+cu" in torch.__version__:
        build = "CUDA"
    elif "+rocm" in torch.__version__:
        build = "ROCm"

    if build == "CUDA":
        print(
            "BUILD       : CUDA wheel — WRONG ON THIS BOX. There is no NVIDIA "
            "hardware here; this build will run CPU-only forever while looking "
            "healthy. Rebuild the venv from scratch (see pyproject.toml)."
        )
        return 1

    print(f"build       : {build}")

    # ROCm masquerades as CUDA in the torch API; `device='cuda'` is the correct
    # call on this box and there are no ROCm-specific device strings.
    available = torch.cuda.is_available()
    device = "cuda" if available else "cpu"
    print(f"accelerator : {'yes — ' + torch.cuda.get_device_name(0) if available else 'no (CPU)'}")
    print(f"device      : {device}")
    print(f"threads     : {torch.get_num_threads()}")

    if args.matmul_n > 0:
        n = args.matmul_n
        a = torch.randn(n, n, device=device)
        b = torch.randn(n, n, device=device)
        for _ in range(3):  # warm-up
            a @ b
        if available:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        reps = 10
        for _ in range(reps):
            a @ b
        if available:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        tflops = (2.0 * n**3 * reps) / elapsed / 1e12
        print(f"matmul {n}^3 : {tflops:.2f} TFLOP/s fp32 on {device}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
