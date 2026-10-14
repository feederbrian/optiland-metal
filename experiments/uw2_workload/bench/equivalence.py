"""Bit-for-bit equivalence check for optimizations, on the UW2 lens.

Runs a fixed sequence of lens states shaped like the optimizer's work (a
base point, one-sided finite-difference steps on variables before and after
the stop, a return to base, an accepted multi-variable step) and records
every operand value plus raw image-plane ray data at each state. The same
sequence runs against two Optiland trees and the outputs must be identical,
NaNs included:

    # stock reference (a checkout of upstream-master)
    PYTHONPATH=/path/to/stock .venv/bin/python \\
        experiments/uw2_workload/bench/equivalence.py --out stock.npz
    # the tree under test
    .venv/bin/python experiments/uw2_workload/bench/equivalence.py --out new.npz
    .venv/bin/python experiments/uw2_workload/bench/equivalence.py \\
        --compare stock.npz new.npz

Because the robust aimer warm-starts from the previous state's pupil maps,
an operand's last bits depend on the call history (about 1e-11 waves on this
lens). Both runs replay the identical history, so any difference is the
optimization's.
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings

import numpy as np
import uw1_optic
import uw2_operands as ops

warnings.filterwarnings("ignore")

FD_REL = 1e-5  # the job's finite-difference step, applied relatively here
RAY_FIELDS = ("x", "y", "z", "L", "M", "N", "i", "opd")


def _variables(lens):
    """(name, getter, setter) for a spread of variable kinds."""
    s = lens.surfaces.surfaces

    def radius(i):
        return (
            f"R{i}",
            lambda: float(s[i].geometry.radius),
            lambda v: lens.updater.set_radius(v, i),
        )

    def thickness(i):
        return (
            f"T{i}",
            lambda: ops._thickness(lens, i),
            lambda v: lens.updater.set_thickness(v, i),
        )

    def asphere(i, j):
        def get():
            return float(s[i].geometry.coefficients[j])

        def put(v):
            # Writes into the live coefficient array, as the optimizer does.
            lens.updater.set_asphere_coeff(v, i, j)

        return (f"A{i}.{j}", get, put)

    return [
        radius(3),  # before the stop: pupil maps go stale
        asphere(1, 1),  # even asphere before the stop
        thickness(7),
        radius(20),  # after the stop: maps stay fresh
        thickness(24),
        thickness(15),  # air gap into the stop
    ]


def _rays(lens, u, wl):
    rays = lens.trace(0.0, u, wl, 6, "hexapolar", record=False)
    return {f: ops._np(getattr(rays, f)) for f in RAY_FIELDS}


def _quick(lens) -> dict[str, np.ndarray]:
    out = {
        "wave": np.array(
            [ops.wave_rms(lens, 0.7, 0.4861), ops.wave_rms(lens, 1.0, 0.6563)]
        ),
        "spot": np.array([ops.spot_rms(lens, 1.0, wls=(0.4861, 0.5876))]),
        "chief": np.array(ops.chief(lens, 0.9)),
    }
    for name, value in _rays(lens, 0.7, ops.D_LINE).items():
        out[f"rays.{name}"] = value
    return out


def run(optic=None, full: bool = True) -> dict[str, np.ndarray]:
    """Replay the state sequence; return {state/quantity: values}."""
    lens = uw1_optic.load(optic)
    results: dict[str, np.ndarray] = {}

    def record(state, values):
        for key, value in values.items():
            results[f"{state}/{key}"] = np.asarray(value, dtype=np.float64)

    if full:
        record("base.full", ops.evaluate(lens))
    record("base", _quick(lens))
    for name, get, put in _variables(lens):
        x0 = get()
        put(x0 * (1 + FD_REL) if x0 else FD_REL)
        try:
            record(f"fd.{name}", _quick(lens))
        finally:
            put(x0)
    record("base.again", _quick(lens))

    # An accepted step: several variables move at once, then stay.
    for _name, get, put in _variables(lens)[:4]:
        x0 = get()
        put(x0 * (1 + 3e-4) if x0 else 1e-4)
    record("step", _quick(lens))
    if full:
        record("step.full", ops.evaluate(lens))
    return results


def compare(path_a: str, path_b: str) -> bool:
    """Report per-quantity differences; True when everything is identical."""
    a, b = np.load(path_a), np.load(path_b)
    keys_a, keys_b = set(a.files), set(b.files)
    same = keys_a == keys_b
    if not same:
        print("key sets differ:", sorted(keys_a ^ keys_b))
    worst = 0.0
    n_diff = 0
    for key in sorted(keys_a & keys_b):
        x, y = a[key], b[key]
        if x.shape != y.shape:
            print(f"{key}: shape {x.shape} vs {y.shape}")
            same = False
            continue
        if np.array_equal(x, y, equal_nan=True):
            continue
        same = False
        n_diff += 1
        finite = np.isfinite(x) & np.isfinite(y)
        nan_mismatch = int(np.sum(np.isnan(x) != np.isnan(y)))
        diff = float(np.max(np.abs(x[finite] - y[finite]))) if finite.any() else 0.0
        worst = max(worst, diff)
        print(f"{key}: max |diff| {diff:.3g}, NaN mismatches {nan_mismatch}")
    total = len(keys_a & keys_b)
    elements = sum(a[k].size for k in keys_a & keys_b)
    if same:
        print(f"identical: {total} quantities, {elements} values, NaNs included")
    else:
        print(f"{n_diff} of {total} quantities differ; worst |diff| {worst:.3g}")
    return same


def main():
    """CLI: record a run or compare two."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--optic", default=None)
    parser.add_argument("--out")
    parser.add_argument("--quick", action="store_true", help="skip full merits")
    parser.add_argument("--compare", nargs=2, metavar=("A", "B"))
    args = parser.parse_args()
    if args.compare:
        sys.exit(0 if compare(*args.compare) else 1)
    if not args.out:
        parser.error("--out or --compare is required")

    import optiland

    print(f"optiland from {optiland.__file__}")
    start = time.perf_counter()
    results = run(args.optic, full=not args.quick)
    elapsed = time.perf_counter() - start
    np.savez(args.out, **results)
    print(f"{len(results)} quantities in {elapsed:.1f} s -> {args.out}")


if __name__ == "__main__":
    main()
