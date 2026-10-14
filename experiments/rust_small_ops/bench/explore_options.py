"""Options that change how many evaluations run, or where they run.

    .venv/bin/python experiments/rust_small_ops/bench/explore_options.py

* Optimizer choice: Optiland's ``LeastSquares`` (scipy ``least_squares``,
  trust-region-reflective) against the tutorial's ``OptimizerGeneric``
  (L-BFGS-B), on stock Optiland and on the Rust compiled lens.
* Process-parallel finite differences: the N+1 evaluations of each gradient
  spread over worker processes, each holding its own stock Optiland lens.

Every final design is re-scored with Optiland's own merit function.
Appends to ``results/small_ops.json`` under ``"options"``.
"""

from __future__ import annotations

import os

# See run_small_ops.py: keep OpenBLAS from oversubscribing the cores that
# worker processes and parallel kernels use. Must precede the NumPy import.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import json
import math
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import patches
import snapshot
import workload as common
from scipy import optimize

from optiland import optimization

RESULTS = Path(__file__).resolve().parents[1] / "results"
WORKERS = os.cpu_count() or 1
TOL = 1e-9


def rescore(x) -> float:
    """Optiland's merit (sum of squares) at scaled vector ``x``."""
    lens = common.tutorial_triplet()
    problem = common.tutorial_problem(lens)
    return float(optimization.OptimizerGeneric(problem)._fun(np.asarray(x, np.float64)))


def report(name, wall, calls, result) -> dict:
    """Print and record one run."""
    x = np.asarray(result.x, np.float64)
    merit = rescore(x)
    print(
        f"  {name:58s} {wall:8.2f} s  {calls:6d} evals  merit {merit:.6g}",
        flush=True,
    )
    return {"wall_seconds": wall, "merit_evaluations": calls, "rescored_merit": merit}


# --------------------------------------------------------------------------
# Least squares
# --------------------------------------------------------------------------


def lsq_optiland(with_fixes: bool) -> dict:
    """``LeastSquares(problem).optimize(method_choice="trf", tol=1e-9)``."""
    lens = common.tutorial_triplet()
    problem = common.tutorial_problem(lens)
    opt = optimization.LeastSquares(problem)
    calls = {"n": 0}
    original = opt._compute_residuals_vector

    def counted(x):
        calls["n"] += 1
        return original(x)

    opt._compute_residuals_vector = counted
    ctx = patches.evaluation_scope() if with_fixes else _null()
    backend = patches.direct_backend() if with_fixes else _null()
    with ctx, backend:
        start = time.perf_counter()
        result = opt.optimize(tol=TOL, method_choice="trf")
        wall = time.perf_counter() - start
    name = "Optiland LeastSquares (trf)" + (" + Python fixes" if with_fixes else "")
    return report(name, wall, calls["n"], result)


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _residual_penalty(m: int) -> np.ndarray:
    return np.full(m, math.sqrt(1e10 / m))


def lsq_rust(batched_jacobian: bool) -> dict:
    """The same ``least_squares`` call on the Rust compiled lens."""
    lens = common.tutorial_triplet()
    problem = common.tutorial_problem(lens)
    rust = snapshot.to_rust(snapshot.capture(lens, problem))
    x0 = np.array([float(v.value) for v in problem.variables])
    bounds = [tuple(v.bounds) for v in problem.variables]
    lower = np.array([b[0] if b[0] is not None else -np.inf for b in bounds])
    upper = np.array([b[1] if b[1] is not None else np.inf for b in bounds])
    m = len(problem.operands)
    calls = {"n": 0}

    def residuals(x):
        calls["n"] += 1
        r = np.asarray(rust.residuals_at(list(x)))
        return _residual_penalty(m) if np.isnan(r).any() else r

    kwargs = {}
    if batched_jacobian:
        # scipy's '2-point' scheme: h = sqrt(eps) * max(1, |x|) * sign(x),
        # flipped inward at a bound; all probes in one parallel Rust call.
        rel = np.finfo(np.float64).eps ** 0.5

        def jacobian(x):
            sign = np.where(x >= 0, 1.0, -1.0)
            h = rel * sign * np.maximum(1.0, np.abs(x))
            h = np.where((x + h > upper) | (x + h < lower), -h, h)
            h = (x + h) - x
            rows = np.vstack([x, x + np.diag(h)])
            calls["n"] += len(rows)
            values = rust.residuals_many(rows)
            if np.isnan(values).any():
                values = np.where(np.isnan(values), math.sqrt(1e10 / m), values)
            return ((values[1:] - values[0]) / h[:, None]).T

        kwargs["jac"] = jacobian

    start = time.perf_counter()
    result = optimize.least_squares(
        residuals, x0, method="trf", bounds=(lower, upper), ftol=TOL, **kwargs
    )
    wall = time.perf_counter() - start
    name = "Rust compiled lens, least_squares (trf)" + (
        f", batched parallel Jacobian ({WORKERS} threads)" if batched_jacobian else ""
    )
    return report(name, wall, calls["n"], result)


# --------------------------------------------------------------------------
# Process-parallel finite differences on stock Optiland
# --------------------------------------------------------------------------

_WORKER: dict = {}


def _init_worker(with_fixes: bool):
    lens = common.tutorial_triplet()
    problem = common.tutorial_problem(lens)
    _WORKER["opt"] = optimization.OptimizerGeneric(problem)
    if with_fixes:
        _WORKER["fixes"] = [patches.evaluation_scope(), patches.direct_backend()]
        for ctx in _WORKER["fixes"]:
            ctx.__enter__()


def _worker_merit(x) -> float:
    return float(_WORKER["opt"]._fun(np.asarray(x, np.float64)))


def fd_parallel_optiland(with_fixes: bool) -> dict:
    """L-BFGS-B with each gradient's N+1 evaluations spread over processes."""
    lens = common.tutorial_triplet()
    problem = common.tutorial_problem(lens)
    x0 = np.array([float(v.value) for v in problem.variables])
    bounds = [tuple(v.bounds) for v in problem.variables]
    calls = {"n": 0}

    from run_small_ops import fd_rows  # noqa: PLC0415

    context = multiprocessing.get_context("forkserver")
    with ProcessPoolExecutor(
        WORKERS, mp_context=context, initializer=_init_worker, initargs=(with_fixes,)
    ) as pool:
        list(pool.map(_worker_merit, [x0] * WORKERS))  # warm every worker

        def value_and_grad(x):
            rows, steps = fd_rows(x, bounds)
            calls["n"] += len(rows)
            values = np.array(list(pool.map(_worker_merit, rows)))
            if np.isnan(values[0]):
                return 1e10, np.zeros_like(x)
            return values[0], (values[1:] - values[0]) / steps

        start = time.perf_counter()
        result = optimize.minimize(
            value_and_grad,
            x0,
            jac=True,
            bounds=bounds,
            tol=TOL,
            options={"maxiter": 1000, "disp": False},
        )
        wall = time.perf_counter() - start
    name = f"Optiland L-BFGS-B, gradients over {WORKERS} processes" + (
        " + Python fixes" if with_fixes else ""
    )
    return report(name, wall, calls["n"], result)


def main():
    """Run every option and merge the results into ``small_ops.json``."""
    out = {}
    print("Optimizer choice (same merit function, same start):")
    out["lsq_optiland"] = lsq_optiland(with_fixes=False)
    out["lsq_optiland_fixes"] = lsq_optiland(with_fixes=True)
    out["lsq_rust"] = lsq_rust(batched_jacobian=False)
    out["lsq_rust_batched"] = lsq_rust(batched_jacobian=True)
    print("\nProcess-parallel finite differences:")
    out["fd_processes"] = fd_parallel_optiland(with_fixes=False)
    out["fd_processes_fixes"] = fd_parallel_optiland(with_fixes=True)

    path = RESULTS / "small_ops.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    data["options"] = out
    path.write_text(json.dumps(data, indent=2))
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
