"""Halt at the first checkpoint that cannot hold §8's floor — minutes, not hours.

Two halts, both DERIVED rather than invented.

**The iteration-0 tripwire.** `ckpt-0` must be bit-identical to the sweep's
`f = 0.65`, seed 0 arm. The sweep trained from the anchor on ring-0; the driver's
iteration 0 does the same thing by a different path, and these are **two
implementations of training that have never been compared** — the harness that
SELECTED the treatment and the driver that APPLIES it. The incoming mode differs
(the driver calls `.eval()` after loading, the sweep does not) and cannot explain
a mismatch: `module.training` is not in `state_dict()`, so it cannot reach a
parameter digest even in principle. **A digest difference has no benign reading.**

**The per-checkpoint band.** Top-1 is read at EVERY checkpoint, not only at
iteration 4. The band's derivation — top-1 → at-par at 2.76 points per point →
the floor — **never references an iteration count**, so it applies at every
checkpoint. A `ckpt-i` below 0.968 cannot hold `at-par >= 1188` at the cadence,
and continuing to iteration 4 spends hours confirming what is already known.

§8 answers the obvious objection in advance: the loop "does not trade one
budget's competence for another's in either direction", so a model legitimately
learning cannot buy scripted competence with suite competence. This halt comes
from the frozen page, not from impatience.

Attempt 1's trajectory is the comparison line:
``0.8942 / 0.8942 / 0.8877 / 0.8893 / 0.8845``. The expectation is ~0.9699
throughout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import time
from pathlib import Path

import torch

from reckoner.config import Config, validate
from reckoner.model import load_checkpoint
from reckoner.train import SupervisionSet

REPO = Path(__file__).resolve().parents[1]

BAND = 0.968
ANCHOR_TOP1 = 0.9699
ATTEMPT_1 = (0.8942, 0.8942, 0.8877, 0.8893, 0.8845)


def weight_digest(model) -> str:
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def read_checkpoint(path: Path, cfg, held, shared, top_k_by_depth) -> tuple[float, int, int, str]:
    """``(top1, hits, n, digest)`` for one checkpoint."""
    model, _ = load_checkpoint(path, cfg)
    model.eval()
    clean = top_k_by_depth(model, held, cfg, device="cpu", exclude=shared)
    top1, n = clean["depth_le_3_top1"], clean["depth_le_3_n"]
    return top1, round(top1 * n), n, weight_digest(model)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--pid", type=int, help="the campaign process, killed BY PID")
    parser.add_argument("--expect-ckpt0", help="the sweep arm's digest, quoted from the record")
    parser.add_argument("--until", type=int, default=4)
    parser.add_argument(
        "--validate",
        nargs=2,
        metavar=("BELOW_BAND", "ABOVE_BAND"),
        help="two checkpoints: the first MUST trip the band, the second MUST NOT. "
        "This component holds authority to kill a six-hour job, so it is shown to "
        "kill correctly and to hold its fire correctly before it is trusted with "
        "one — the same standard every other detector met.",
    )
    args = parser.parse_args()

    from reckoner.dataset import anchored_data
    from scripts.train_phase1 import shared_state_indices, top_k_by_depth

    cfg = Config()
    validate(cfg)
    torch.set_num_threads(cfg.campaign.intra_op_threads)
    eval_path = anchored_data("phase1_eval")
    held = SupervisionSet(eval_path)
    shared = shared_state_indices(eval_path)

    if args.validate:
        below, above = (Path(p) for p in args.validate)
        print(f"\n  WATCHER VALIDATION — band {BAND}\n", flush=True)
        ok = True
        for path, must_fire in ((below, True), (above, False)):
            top1, hits, n, _ = read_checkpoint(path, cfg, held, shared, top_k_by_depth)
            fires = top1 < BAND
            good = fires == must_fire
            ok &= good
            print(
                f"    {'ok  ' if good else 'FAIL'} {path.name:<14} top-1 {top1:.4f} "
                f"({hits}/{n})  fires={fires}  required={must_fire}",
                flush=True,
            )
        print(
            f"\n  {'VALIDATED' if ok else 'NOT VALIDATED'} — the killer "
            f"{'kills and holds fire correctly' if ok else 'CANNOT BE TRUSTED'}.",
            flush=True,
        )
        return 0 if ok else 1

    if args.pid is None or args.expect_ckpt0 is None:
        parser.error("--pid and --expect-ckpt0 are required unless --validate is given")

    status = args.run_dir / "checkpoint_watch.json"
    print(f"\n  CHECKPOINT WATCH — band {BAND}, anchor {ANCHOR_TOP1}", flush=True)
    print(f"  attempt 1 read {' / '.join(f'{v:.4f}' for v in ATTEMPT_1)}", flush=True)
    print("  reading at the COMMIT POINT: ckpt-i is examined only once LATEST >= i,", flush=True)
    print("  never on file appearance — a checkpoint caught mid-write yields a", flush=True)
    print("  garbage top-1, and a garbage top-1 yields a wrongful kill.\n", flush=True)

    seen: set[int] = set()
    verdicts: list[dict] = []
    halted = False

    def record() -> None:
        status.write_text(
            json.dumps(
                {
                    "band": BAND,
                    "expected_ckpt0": args.expect_ckpt0,
                    "attempt_1": list(ATTEMPT_1),
                    "verdicts": verdicts,
                    "halted": halted,
                },
                indent=2,
            )
            + "\n"
        )

    record()
    while not halted:
        if not Path(f"/proc/{args.pid}").exists():
            print("  campaign process exited; watch ends.", flush=True)
            break
        marker = args.run_dir / "LATEST"
        latest = int(marker.read_text().strip()) if marker.exists() else -1
        for index in range(latest + 1):
            if index in seen:
                continue
            path = args.run_dir / f"ckpt-{index}.pt"
            if not path.exists():
                continue
            top1, hits, n, digest = read_checkpoint(path, cfg, held, shared, top_k_by_depth)
            seen.add(index)
            holds = top1 >= BAND
            verdicts.append(
                {
                    "iteration": index,
                    "top1": top1,
                    "hits": hits,
                    "n": n,
                    "digest": digest,
                    "holds_band": holds,
                }
            )

            if index == 0:
                match = digest == args.expect_ckpt0
                print(
                    f"  TRIPWIRE ckpt-0 digest {digest[:16]} "
                    f"{'==' if match else '!='} sweep arm {args.expect_ckpt0[:16]}",
                    flush=True,
                )
                if not match:
                    print("\n  HALT — the driver's iteration 0 is NOT the sweep's arm.", flush=True)
                    print(
                        "  `module.training` is not in state_dict, so the mode difference",
                        flush=True,
                    )
                    print(
                        "  cannot reach a parameter digest: there is no benign reading.", flush=True
                    )
                    halted = True

            print(
                f"  {'ok  ' if holds else 'HALT'} ckpt-{index}: top-1 {top1:.4f} "
                f"({hits}/{n})   attempt 1 read {ATTEMPT_1[index]:.4f}",
                flush=True,
            )
            if not holds:
                print(
                    f"\n  HALT — ckpt-{index} is below the band, so the cadence cannot", flush=True
                )
                print(
                    "  hold at-par >= 1188. The band's derivation never references an", flush=True
                )
                print("  iteration count, so it applies here.", flush=True)
                halted = True
            record()
            if halted:
                os.kill(args.pid, signal.SIGKILL)
                break
        if len(seen) > args.until:
            print("\n  every checkpoint held the band.", flush=True)
            break
        time.sleep(20)

    record()
    return 1 if halted else 0


if __name__ == "__main__":
    raise SystemExit(main())
