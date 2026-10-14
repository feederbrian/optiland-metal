"""Time single proxy operands and one merit evaluation on the UW2 lens.

    .venv/bin/python experiments/uw2_workload/bench/bench_operands.py \\
        --out results/new_operands.json
    PYTHONPATH=/path/to/stock .venv/bin/python ... --out results/stock_operands.json

"perturbed" changes a radius ahead of the stop by the job's relative step
before every call, as inside a Jacobian, so the aimer's pupil maps are stale
each time; "fresh" repeats the call on an unchanged lens.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings

import numpy as np
import uw1_optic
import uw2_operands as ops

warnings.filterwarnings("ignore")

OPERANDS = {
    "wave (u=0.7, F line)": lambda lens: ops.wave_rms(lens, 0.7, 0.4861),
    "chief (u=1)": lambda lens: ops.chief(lens, 1.0),
    "spot (u=0.7, 5 wavelengths)": lambda lens: ops.spot_rms(lens, 0.7),
}


def _time(lens, fn, perturbed: bool, repeats: int) -> float:
    r0 = float(lens.surfaces.surfaces[3].geometry.radius)
    fn(lens)
    times = []
    for k in range(repeats):
        if perturbed:
            lens.updater.set_radius(r0 * (1 + 1e-5 * (k % 2 + 1)), 3)
        start = time.perf_counter()
        fn(lens)
        times.append(time.perf_counter() - start)
    lens.updater.set_radius(r0, 3)
    return float(np.median(times))


def main():
    """Time each operand fresh and perturbed, then a perturbed merit."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--optic", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--repeats", type=int, default=15)
    args = parser.parse_args()

    import optiland

    print(f"optiland from {optiland.__file__}")
    out = {}
    for name, fn in OPERANDS.items():
        for mode in ("fresh", "perturbed"):
            lens = uw1_optic.load(args.optic)
            seconds = _time(lens, fn, mode == "perturbed", args.repeats)
            out[f"{name} [{mode}]"] = seconds
            print(f"{name} [{mode}]: {seconds * 1e3:.1f} ms")

    lens = uw1_optic.load(args.optic)
    ops.evaluate(lens)
    r0 = float(lens.surfaces.surfaces[3].geometry.radius)
    merits = []
    for k in range(3):
        lens.updater.set_radius(r0 * (1 + 1e-5 * (k + 1)), 3)
        start = time.perf_counter()
        ops.evaluate(lens)
        merits.append(time.perf_counter() - start)
    out["merit [perturbed]"] = float(np.median(merits))
    print(f"merit [perturbed]: {out['merit [perturbed]']:.2f} s")
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
