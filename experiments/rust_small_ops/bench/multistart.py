"""What the speed buys: multi-start optimization in the time of one stock run.

The tutorial's single L-BFGS-B run lands in one of many local minima (tiny
round-off differences already change which one, see the free-running runs).
With the compiled lens one run takes about a second, so many starts fit in
the time of a single stock run.

    .venv/bin/python experiments/rust_small_ops/bench/multistart.py [--starts 32]

Appends ``"multistart"`` to ``results/small_ops.json``.
"""

from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # see run_small_ops.py

import argparse
import json
import time
from pathlib import Path

import numpy as np
from run_small_ops import THREADS, Workload, _batched_gradient, _scipy_run

RESULTS = Path(__file__).resolve().parents[1] / "results"


def main():
    """Run perturbed starts with the Numba lens and a parallel gradient."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--starts", type=int, default=32)
    parser.add_argument("--sigma", type=float, default=0.1, help="scaled units")
    args = parser.parse_args()

    w = Workload()
    lower = np.array([b[0] for b in w.bounds])
    upper = np.array([b[1] for b in w.bounds])
    rng = np.random.default_rng(0)
    objective = _batched_gradient(w.numba.merit_many, w.bounds)
    objective(w.x0)  # compile outside the timed region

    finals = []
    start = time.perf_counter()
    for k in range(args.starts):
        x0 = w.x0 if k == 0 else w.x0 + rng.normal(scale=args.sigma, size=w.x0.size)
        x0 = np.clip(x0, lower + 1e-6, upper - 1e-6)
        result, _, _ = _scipy_run(objective, x0, w.bounds, jac=True)
        finals.append((float(result.fun), np.asarray(result.x, np.float64)))
    wall = time.perf_counter() - start

    merits = np.array([f for f, _ in finals])
    best_merit, best_x = min(finals, key=lambda fx: fx[0])
    rescored = w.optiland_merit(best_x)
    out = {
        "starts": args.starts,
        "sigma": args.sigma,
        "threads": THREADS,
        "wall_seconds": wall,
        "best_merit": best_merit,
        "best_rescored_by_optiland": rescored,
        "median_merit": float(np.median(merits)),
        "share_better_than_stock": None,
        "best_x": best_x.tolist(),
    }
    stock_path = RESULTS / "characterize.json"
    if stock_path.exists():
        stock = json.loads(stock_path.read_text())["optimize"]["final_merit"]
        out["stock_final_merit"] = stock
        out["share_better_than_stock"] = float(np.mean(merits < stock))
    print(
        f"{args.starts} starts in {wall:.1f} s: best merit {best_merit:.6g} "
        f"(Optiland re-score {rescored:.6g}), median {np.median(merits):.4g}"
    )
    if out["share_better_than_stock"] is not None:
        print(
            f"{out['share_better_than_stock']:.0%} of starts beat the stock run's "
            f"{out['stock_final_merit']:.4g}"
        )

    path = RESULTS / "small_ops.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    data["multistart"] = out
    path.write_text(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
