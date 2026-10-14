"""Profile one proxy operand on the UW2 lens, the way the optimizer sees it.

    .venv/bin/python experiments/uw2_workload/bench/profile_operand.py
        [--optic PATH] [--operand wave|chief|spot] [--mode fresh|perturbed]
        [--top 30]

``perturbed`` changes a pre-stop radius by 1e-5 (the job's FD step) before
every call, so the robust aimer's pupil maps go stale as they do inside a
Jacobian; ``fresh`` repeats the call on an unchanged lens.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import time
import warnings

import uw1_optic
import uw2_operands as ops

warnings.filterwarnings("ignore")

OPERANDS = {
    "wave": lambda lens: ops.wave_rms(lens, 0.7, 0.4861),
    "chief": lambda lens: ops.chief(lens, 1.0),
    "spot": lambda lens: ops.spot_rms(lens, 0.7),
}


def main():
    """Time and profile the chosen operand."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--optic", default=None)
    parser.add_argument("--operand", choices=OPERANDS, default="wave")
    parser.add_argument("--mode", choices=("fresh", "perturbed"), default="perturbed")
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()

    lens = uw1_optic.load(args.optic)
    fn = OPERANDS[args.operand]
    fn(lens)  # warm: JIT, glass tables, first pupil maps
    r0 = float(lens.surfaces.surfaces[3].geometry.radius)

    def call(k):
        if args.mode == "perturbed":
            lens.updater.set_radius(r0 * (1 + 1e-5 * (k % 2 + 1)), 3)
        return fn(lens)

    start = time.perf_counter()
    for k in range(3):
        call(k)
    per_call = (time.perf_counter() - start) / 3
    print(f"{args.operand} ({args.mode}): {per_call * 1e3:.0f} ms per call")

    profiler = cProfile.Profile()
    profiler.enable()
    for k in range(3, 6):
        call(k)
    profiler.disable()
    out = io.StringIO()
    stats = pstats.Stats(profiler, stream=out)
    print(f"  {stats.total_calls / 3:,.0f} Python calls per call")
    stats.sort_stats("cumulative").print_stats(args.top)
    for line in out.getvalue().splitlines():
        if line.strip() and "ncalls" not in line[:12] or "ncalls" in line:
            print(line.replace("/home/user/optiland-upstream/", "")[:170])


if __name__ == "__main__":
    main()
