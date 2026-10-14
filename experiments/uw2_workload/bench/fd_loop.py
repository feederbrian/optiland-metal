"""Time a finite-difference Jacobian loop on the UW2 lens, the pipeline's hot loop.

Each column perturbs one variable by the job's relative step, evaluates the
full proxy merit (``uw2_operands.evaluate``) and restores the variable, as
one Jacobian worker does. Every operand value is saved, so two runs can be
compared bit for bit with ``equivalence.py --compare``:

    PYTHONPATH=/path/to/stock .venv/bin/python \\
        experiments/uw2_workload/bench/fd_loop.py --out stock_fd.npz
    .venv/bin/python experiments/uw2_workload/bench/fd_loop.py --out new_fd.npz
    .venv/bin/python experiments/uw2_workload/bench/equivalence.py \\
        --compare stock_fd.npz new_fd.npz
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

FD_REL = 1e-5


def _variables(lens):
    """A spread of the job's variable kinds, before and after the stop (16)."""
    s = lens.surfaces.surfaces
    out = []
    for i in (1, 3, 12, 20):
        out.append(
            (
                f"R{i}",
                lambda i=i: float(s[i].geometry.radius),
                lambda v, i=i: lens.updater.set_radius(v, i),
            )
        )
    for i in (7, 15, 24):
        out.append(
            (
                f"T{i}",
                lambda i=i: ops._thickness(lens, i),
                lambda v, i=i: lens.updater.set_thickness(v, i),
            )
        )
    out.append(
        (
            "A1.1",
            lambda: float(s[1].geometry.coefficients[1]),
            lambda v: lens.updater.set_asphere_coeff(v, 1, 1),
        )
    )
    return out


def main():
    """Run the loop; print per-column times; save values and timings."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--optic", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--columns", type=int, default=8)
    args = parser.parse_args()

    import optiland

    print(f"optiland from {optiland.__file__}")
    lens = uw1_optic.load(args.optic)
    results: dict[str, np.ndarray] = {}
    timings: dict[str, float] = {}

    start = time.perf_counter()
    base = ops.evaluate(lens)
    timings["base"] = time.perf_counter() - start
    for key, value in base.items():
        results[f"base/{key}"] = value
    print(f"base merit {timings['base']:.2f} s")

    for name, get, put in _variables(lens)[: args.columns]:
        x0 = get()
        put(x0 * (1 + FD_REL))
        start = time.perf_counter()
        try:
            values = ops.evaluate(lens)
        finally:
            put(x0)
        timings[name] = time.perf_counter() - start
        for key, value in values.items():
            results[f"{name}/{key}"] = value
        print(f"column {name}: {timings[name]:.2f} s")

    columns = [v for k, v in timings.items() if k != "base"]
    print(f"mean column {np.mean(columns):.2f} s over {len(columns)}")
    np.savez(args.out, **results)
    with open(args.out.removesuffix(".npz") + ".json", "w") as fh:
        json.dump(timings, fh, indent=1)


if __name__ == "__main__":
    main()
