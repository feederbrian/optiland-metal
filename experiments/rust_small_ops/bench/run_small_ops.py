"""Can Rust outperform Optiland on small, overhead-bound operations?

    .venv/bin/python experiments/rust_small_ops/bench/run_small_ops.py
        [--only boundary,leaf,patches,compiled,gradient,optimize,blas]
        [--skip-slow]

Writes ``results/small_ops.json`` (render with ``report.py``). Every
implementation is checked against Optiland's own result before timing, and
every optimization result is re-scored with Optiland's merit function.
"""

from __future__ import annotations

import os

# OpenBLAS (used by scipy's optimizers) otherwise starts one spinning thread
# per core, which steals the cores the parallel Rust/Numba kernels need. It
# must be set before NumPy is imported. Serial timings are unaffected.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import contextlib
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import lean
import numba
import numba_lens
import numpy as np
import optiland_small
import patches
import snapshot
import workload as common
from numba import njit
from scipy import optimize

import optiland.backend as be
from optiland import optimization
from optiland.optimization.operand.ray import RayOperand

RESULTS = Path(__file__).resolve().parents[1] / "results"
THREADS = os.cpu_count() or 1
SECTIONS = ("boundary", "leaf", "patches", "compiled", "gradient", "optimize", "blas")
WL_INDEX = 1  # primary wavelength of the tutorial triplet
RAY = (0.0, 1.0, 0.3, 0.8)  # Hx, Hy, Px, Py for the single-ray probes


def timed(label: str, fn, repeats: int = 5) -> float:
    """Per-call seconds with a progress line."""
    seconds = common.per_call(fn, repeats=repeats)
    print(f"    {label:52s} {common.fmt(seconds):>10s}", flush=True)
    return seconds


@contextlib.contextmanager
def python_fixes():
    """The two pure-Python Optiland fixes together."""
    with patches.evaluation_scope(), patches.direct_backend():
        yield


class Workload:
    """The tutorial problem plus every compiled-lens implementation of it."""

    def __init__(self):
        self.lens = common.tutorial_triplet()
        self.problem = common.tutorial_problem(self.lens)
        self.optimizer = optimization.OptimizerGeneric(self.problem)
        self.x0 = np.array([float(v.value) for v in self.problem.variables])
        self.bounds = [tuple(v.bounds) for v in self.problem.variables]
        self.snap = snapshot.capture(self.lens, self.problem)
        self.rust = snapshot.to_rust(self.snap)
        self.numba = numba_lens.NumbaLens(self.snap)
        self.python = lean.PythonLens(self.snap)
        self.numpy = lean.NumpyLens(self.snap)
        # Two nearby designs; alternating between them defeats any caching
        # and mirrors an optimizer changing the lens on every call.
        step = 0.02 * np.linspace(-1, 1, self.x0.size)
        self.xs = (self.x0 + step, self.x0 - step)
        self._k = 0

    def alternate(self):
        self._k ^= 1
        return self.xs[self._k]

    def optiland_merit(self, x) -> float:
        return float(self.optimizer._fun(np.asarray(x, np.float64)))


# --------------------------------------------------------------------------
# A. boundary cost
# --------------------------------------------------------------------------


@njit(cache=True)
def _nb_noop():
    pass


@njit(cache=True)
def _nb_add3(a, b, c):
    return a + b + c


@njit(cache=True)
def _nb_echo8(a, b, c, d, e, f, g, h):
    return (
        a.copy(),
        b.copy(),
        c.copy(),
        d.copy(),
        e.copy(),
        f.copy(),
        g.copy(),
        h.copy(),
    )


def _py_noop():
    pass


def _py_add3(a, b, c):
    return a + b + c


def bench_boundary(_: Workload) -> dict:
    """What one call costs before any work is done."""
    one = [np.ones(1) for _ in range(8)]
    ray91 = [np.ones(91) for _ in range(8)]
    a1, b1 = np.ones(1), np.ones(1)
    cases = {
        "Python function, no args": _py_noop,
        "Python function, 3 floats": lambda: _py_add3(1.0, 2.0, 3.0),
        "Rust (PyO3), no args": optiland_small.noop,
        "Rust (PyO3), 3 floats": lambda: optiland_small.add3(1.0, 2.0, 3.0),
        "Rust (PyO3), 8 arrays of 1 in, 8 out": lambda: optiland_small.echo_arrays8(
            one
        ),
        "Rust (PyO3), 8 arrays of 91 in, 8 out": lambda: optiland_small.echo_arrays8(
            ray91
        ),
        "Numba, no args": _nb_noop,
        "Numba, 3 floats": lambda: _nb_add3(1.0, 2.0, 3.0),
        "Numba, 8 arrays of 1 in, 8 out": lambda: _nb_echo8(*one),
        "Numba, 8 arrays of 91 in, 8 out": lambda: _nb_echo8(*ray91),
        "NumPy a + b, 1 element": lambda: a1 + b1,
        "NumPy np.sqrt(a), 1 element": lambda: np.sqrt(a1),
        "Optiland be.sqrt(a), 1 element": lambda: be.sqrt(a1),
        "math.sqrt(x), Python float": lambda: math.sqrt(2.0),
    }
    return {name: timed(name, fn) for name, fn in cases.items()}


# --------------------------------------------------------------------------
# B. Rust leaf inside unchanged Optiland
# --------------------------------------------------------------------------


def bench_leaf(w: Workload) -> dict:
    """Tier 2: surface arithmetic in Rust, Optiland orchestration unchanged."""
    lens, wl = w.lens, w.lens.primary_wavelength
    ops = {
        "trace_generic, 1 ray": lambda: lens.trace_generic(*RAY, wl),
        "trace, 91 rays": lambda: lens.trace(0, 1, wl, common.NUM_RINGS, "hexapolar"),
        "merit evaluation (optimizer._fun)": lambda: w.optimizer._fun(w.alternate()),
    }
    reference = w.optiland_merit(w.xs[0])
    out = {}
    for label, ctx in (
        ("stock", contextlib.nullcontext),
        ("rust leaf", patches.rust_surface_leaf),
    ):
        with ctx():
            assert w.optiland_merit(w.xs[0]) == reference
            out[label] = {
                n: timed(f"{label}: {n}", fn, repeats=3) for n, fn in ops.items()
            }
    return out


def bench_patches(w: Workload) -> dict:
    """Each Optiland prototype alone and combined; results must not change."""
    lens, wl = w.lens, w.lens.primary_wavelength
    reference = [w.optiland_merit(x) for x in w.xs]
    combos = {
        "stock": (),
        "Rust leaf": (patches.rust_surface_leaf,),
        "build once per operation (memo scope)": (patches.evaluation_scope,),
        "direct backend binding": (patches.direct_backend,),
        "memo scope + backend binding": (
            patches.evaluation_scope,
            patches.direct_backend,
        ),
        "all three": (
            patches.evaluation_scope,
            patches.direct_backend,
            patches.rust_surface_leaf,
        ),
    }
    out = {}
    for name, managers in combos.items():
        with contextlib.ExitStack() as stack:
            for manager in managers:
                stack.enter_context(manager())
            assert [w.optiland_merit(x) for x in w.xs] == reference, name
            out[name] = {
                "merit at new x": timed(
                    f"{name}: merit at new x",
                    lambda: w.optimizer._fun(w.alternate()),
                    repeats=3,
                ),
                "trace_generic, 1 ray": timed(
                    f"{name}: trace_generic, 1 ray",
                    lambda: lens.trace_generic(*RAY, wl),
                    repeats=3,
                ),
            }
    return out


_BLAS_PROBE = """
import json, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, {bench!r})
import run_small_ops as r
w = r.Workload()
out = {{}}
for name, many in (("Rust", w.rust.merit_many), ("Numba", w.numba.merit_many)):
    many(w.xs[0][None, :])  # compile / warm
    _, wall, calls = r._scipy_run(
        r._batched_gradient(many, w.bounds), w.x0, w.bounds, jac=True
    )
    out[name] = wall
print(json.dumps(out))
"""


def bench_blas(_: Workload) -> dict:
    """Parallel-gradient optimizations with OpenBLAS at 1 thread vs per core.

    Each setting needs a fresh interpreter: OpenBLAS reads the variable when
    NumPy is first imported.
    """
    out = {}
    bench = str(Path(__file__).resolve().parent)
    for threads in (THREADS, 1):
        env = dict(os.environ, OPENBLAS_NUM_THREADS=str(threads))
        proc = subprocess.run(
            [sys.executable, "-c", _BLAS_PROBE.format(bench=bench)],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        result = json.loads(proc.stdout.strip().splitlines()[-1])
        out[f"OPENBLAS_NUM_THREADS={threads}"] = result
        for name, wall in result.items():
            print(
                f"    OPENBLAS_NUM_THREADS={threads}: {name} + parallel gradient "
                f"{wall:7.2f} s",
                flush=True,
            )
    return out


# --------------------------------------------------------------------------
# C. per-operation latency, every implementation
# --------------------------------------------------------------------------


def _check(name, got, expected, rtol=1e-10):
    assert math.isclose(got, expected, rel_tol=rtol, abs_tol=1e-12), (
        name,
        got,
        expected,
    )


def bench_compiled(w: Workload) -> dict:
    """f2, one ray, one RMS spot and one merit, per implementation."""
    lens, wl = w.lens, w.lens.primary_wavelength
    py, npl, nb, rs = w.python, w.numpy, w.numba, w.rust

    # Earlier sections leave the Optiland lens at a perturbed design; the
    # snapshots hold the starting design, so reset before taking references.
    w.optiland_merit(w.x0)
    expected = {
        "f2": float(lens.paraxial.f2()),
        "ray": float(np.ravel(lens.trace_generic(*RAY, wl).y)[0]),
        "rms": float(
            RayOperand.rms_spot_size(lens, -1, 0, 1.0, common.NUM_RINGS, "all")
        ),
        "merit": w.optiland_merit(w.xs[0]),
    }
    for name, impl in (("python", py), ("numpy", npl)):
        _check(name, impl.f2(), expected["f2"])
        _check(name, impl.rms_spot(impl.epl(), 0, 1.0), expected["rms"])
        _check(name, impl.merit_at(w.xs[0]), expected["merit"])
    _check("python", py.trace_generic(*RAY, WL_INDEX)[1], expected["ray"])
    for name, impl in (("numba", nb), ("rust", rs)):
        _check(name, impl.f2(), expected["f2"])
        _check(name, impl.trace_generic(*RAY, WL_INDEX)[1], expected["ray"])
        _check(name, impl.rms_spot(0.0, 1.0), expected["rms"])
        _check(name, impl.merit_at(list(w.xs[0])), expected["merit"])

    def optiland_ops():
        return {
            "f2": lambda: lens.paraxial.f2(),
            "trace_generic, 1 ray": lambda: lens.trace_generic(*RAY, wl),
            "rms_spot_size, 3 wl x 91 rays": lambda: RayOperand.rms_spot_size(
                lens, -1, 0, 1.0, common.NUM_RINGS, "all"
            ),
            "merit at new x": lambda: w.optimizer._fun(w.alternate()),
        }

    impls = {
        "Optiland (shipped)": (contextlib.nullcontext, optiland_ops()),
        "Optiland + Python fixes": (python_fixes, optiland_ops()),
        "Plain Python, compiled-lens algorithm": (
            contextlib.nullcontext,
            {
                "f2": py.f2,
                "trace_generic, 1 ray": lambda: py.trace_generic(*RAY, WL_INDEX),
                "rms_spot_size, 3 wl x 91 rays": lambda: py.rms_spot(py.epl(), 0, 1.0),
                "merit at new x": lambda: py.merit_at(w.alternate()),
            },
        ),
        "Lean NumPy, compiled-lens algorithm": (
            contextlib.nullcontext,
            {
                "f2": npl.f2,
                "rms_spot_size, 3 wl x 91 rays": lambda: npl.rms_spot(
                    npl.epl(), 0, 1.0
                ),
                "merit at new x": lambda: npl.merit_at(w.alternate()),
            },
        ),
        "Numba": (
            contextlib.nullcontext,
            {
                "f2": nb.f2,
                "trace_generic, 1 ray": lambda: nb.trace_generic(*RAY, WL_INDEX),
                "rms_spot_size, 3 wl x 91 rays": lambda: nb.rms_spot(0.0, 1.0),
                "merit at new x": lambda: nb.merit_at(w.alternate()),
            },
        ),
        "Rust": (
            contextlib.nullcontext,
            {
                "f2": rs.f2,
                "trace_generic, 1 ray": lambda: rs.trace_generic(*RAY, WL_INDEX),
                "rms_spot_size, 3 wl x 91 rays": lambda: rs.rms_spot(0.0, 1.0),
                "merit at new x": lambda: rs.merit_at(list(w.alternate())),
            },
        ),
    }
    out = {}
    for name, (ctx, ops) in impls.items():
        with ctx():
            slow = name.startswith("Optiland") or name.startswith("Plain")
            out[name] = {
                op: timed(f"{name}: {op}", fn, repeats=3 if slow else 5)
                for op, fn in ops.items()
            }
    return out


# --------------------------------------------------------------------------
# D. one finite-difference gradient
# --------------------------------------------------------------------------


def fd_rows(x, bounds, h=1e-8):
    """x and its forward-difference probes, stepping inward at upper bounds."""
    rows, steps = [np.array(x, np.float64)], []
    for i, (_, upper) in enumerate(bounds):
        step = (x[i] + h) - x[i]
        if upper is not None and x[i] + step > upper:
            step = -step
        probe = np.array(x, np.float64)
        probe[i] += step
        rows.append(probe)
        steps.append(step)
    return np.array(rows), np.array(steps)


def bench_gradient(w: Workload) -> dict:
    """N+1 merit evaluations for one forward-difference gradient."""
    rows, _ = fd_rows(w.xs[0], w.bounds)
    expected = np.array([w.optiland_merit(r) for r in rows])
    np.testing.assert_allclose(w.rust.merit_many(rows), expected, rtol=1e-10)
    np.testing.assert_allclose(w.numba.merit_many(rows), expected, rtol=1e-10)

    def numba_threads(n, fn):
        def run():
            previous = numba.get_num_threads()
            numba.set_num_threads(n)
            try:
                return fn()
            finally:
                numba.set_num_threads(previous)

        return run

    label = f"{len(rows)} merit evaluations"
    cases = {
        "Optiland (shipped), sequential": (
            lambda: [w.optimizer._fun(r) for r in rows],
            1,
        ),
        "Plain Python, sequential": (lambda: [w.python.merit_at(r) for r in rows], 3),
        "Numba, 1 thread": (numba_threads(1, lambda: w.numba.merit_many(rows)), 5),
        f"Numba, {THREADS} threads": (
            numba_threads(THREADS, lambda: w.numba.merit_many(rows)),
            5,
        ),
        "Rust, 1 thread": (lambda: w.rust.merit_many(rows, parallel=False), 5),
        f"Rust, {THREADS} threads": (lambda: w.rust.merit_many(rows, parallel=True), 5),
    }
    return {
        name: timed(f"{name}: {label}", fn, repeats=reps)
        for name, (fn, reps) in cases.items()
    }


# --------------------------------------------------------------------------
# E. the whole tutorial optimization
# --------------------------------------------------------------------------


def _scipy_run(objective, x0, bounds, jac=None):
    """``OptimizerGeneric.optimize(tol=1e-9)`` with a different objective."""
    calls = {"n": 0}

    def fun(x):
        calls["n"] += 1
        value = objective(x)
        if jac is True:
            f, g = value
            return (1e10, np.zeros_like(g)) if math.isnan(f) else (f, g)
        return 1e10 if math.isnan(value) else value

    start = time.perf_counter()
    result = optimize.minimize(
        fun,
        x0,
        bounds=bounds,
        jac=jac,
        tol=1e-9,
        options={"maxiter": 1000, "disp": False},
    )
    return result, time.perf_counter() - start, calls["n"]


def _batched_gradient(merit_many, bounds):
    def value_and_grad(x):
        rows, steps = fd_rows(x, bounds)
        values = merit_many(rows)
        return values[0], (values[1:] - values[0]) / steps

    return value_and_grad


def record_trajectory() -> dict:
    """Run the tutorial optimization on Optiland, recording every evaluation.

    Runs with the pure-Python fixes, which reproduce the stock trajectory bit
    for bit (same evaluations, same iterates, same final merit) in a quarter
    of the time. The recorded points let every implementation be timed on
    exactly the same work.
    """
    lens = common.tutorial_triplet()
    problem = common.tutorial_problem(lens)
    opt = optimization.OptimizerGeneric(problem)
    xs, merits = [], []
    original = opt._fun

    def recording(x):
        value = original(x)
        xs.append(np.array(x, np.float64))
        merits.append(float(value))
        return value

    opt._fun = recording
    with python_fixes():
        start = time.perf_counter()
        result = opt.optimize(tol=1e-9, disp=False)
        wall = time.perf_counter() - start
    return {
        "xs": np.array(xs),
        "merits": np.array(merits),
        "wall_seconds": wall,
        "iterations": int(result.nit),
        "final_merit": float(result.fun),
    }


def _replay(name, fn, rows, expected) -> float:
    """Wall time for ``fn`` over the recorded rows; results must match.

    ``OptimizerGeneric._fun`` reports a NaN merit (a failed ray) as 1e10, so
    the recorded values carry that sentinel; NaNs are mapped the same way.
    """
    start = time.perf_counter()
    got = fn(rows)
    wall = time.perf_counter() - start
    got = np.where(np.isnan(got), 1e10, got)
    np.testing.assert_allclose(got, expected, rtol=1e-9, err_msg=name)
    print(f"    {name:52s} {wall:9.2f} s", flush=True)
    return wall


def bench_optimize(w: Workload, skip_slow: bool) -> dict:
    """End to end: same work replayed, then free-running optimizations."""
    out = {"replay": {}, "free_running": {}}
    stock_path = RESULTS / "characterize.json"
    stock = (
        json.loads(stock_path.read_text())["optimize"] if stock_path.exists() else None
    )

    RESULTS.mkdir(exist_ok=True)
    print("  Recording the tutorial trajectory (Optiland + Python fixes)...")
    traj = record_trajectory()
    rows, merits = traj["xs"], traj["merits"]
    n = len(rows)
    if stock:
        # The fixes must not change the optimization at all.
        assert n == stock["merit_evaluations"], (n, stock["merit_evaluations"])
        assert traj["final_merit"] == stock["final_merit"]
        out["replay"]["Optiland (shipped)"] = stock["wall_seconds"]
    out["replay"]["Optiland + Python fixes"] = traj["wall_seconds"]
    best = np.minimum.accumulate(merits)
    out["trajectory"] = {
        "merit_evaluations": n,
        "iterations": traj["iterations"],
        "final_merit": traj["final_merit"],
        # Evaluations the stock run needed before its best merit came within
        # a given fraction of where it finally stopped.
        "evaluations_to_within": {
            f"{frac:g}": int(np.argmax(best <= best[-1] * (1 + frac))) + 1
            for frac in (1.0, 0.1, 0.01, 0.001)
        },
    }
    np.savez_compressed(RESULTS / "trajectory.npz", xs=rows, merits=merits)
    print(
        f"    Optiland + Python fixes (full optimizer run)         "
        f"{traj['wall_seconds']:9.2f} s  ({n} evaluations, {traj['iterations']} iters)"
    )

    # scipy's forward differences evaluate x then its 7 probes back to back,
    # so the trajectory splits into gradient groups of N+1 consecutive rows.
    group = len(w.x0) + 1
    assert n % group == 0
    groups = rows.reshape(-1, group, rows.shape[1])

    def serial(merit_at, as_list=False):
        conv = list if as_list else (lambda r: r)
        return lambda rs: np.array([merit_at(conv(r)) for r in rs])

    def batched(merit_many):
        return lambda rs: np.concatenate([merit_many(g) for g in groups])

    replays = {}
    if not skip_slow:
        replays["Plain Python, compiled-lens algorithm"] = serial(w.python.merit_at)
        replays["Lean NumPy, compiled-lens algorithm"] = serial(w.numpy.merit_at)
    replays["Numba"] = serial(w.numba.merit_at)
    replays["Rust"] = serial(w.rust.merit_at, as_list=True)
    replays[f"Numba, gradient batches in parallel ({THREADS} threads)"] = batched(
        w.numba.merit_many
    )
    replays[f"Rust, gradient batches in parallel ({THREADS} threads)"] = batched(
        w.rust.merit_many
    )
    print(f"  Replaying the same {n:,} evaluations:")
    for name, fn in replays.items():
        out["replay"][name] = _replay(name, fn, rows, merits)

    print("  Free-running optimizations (each follows its own path):")
    runs = {
        "Numba": lambda: _scipy_run(w.numba.merit_at, w.x0, w.bounds),
        "Rust": lambda: _scipy_run(lambda x: w.rust.merit_at(list(x)), w.x0, w.bounds),
        f"Numba + parallel gradient ({THREADS} threads)": lambda: _scipy_run(
            _batched_gradient(w.numba.merit_many, w.bounds), w.x0, w.bounds, jac=True
        ),
        f"Rust + parallel gradient ({THREADS} threads)": lambda: _scipy_run(
            _batched_gradient(w.rust.merit_many, w.bounds), w.x0, w.bounds, jac=True
        ),
    }
    for name, run in runs.items():
        result, wall, calls = run()
        evaluations = calls * (group if "gradient" in name else 1)
        rescored = w.optiland_merit(np.asarray(result.x, np.float64))
        out["free_running"][name] = {
            "wall_seconds": wall,
            "merit_evaluations": evaluations,
            "iterations": int(result.nit),
            "final_merit": float(result.fun),
            "rescored_merit": rescored,
        }
        print(
            f"    {name:52s} {wall:9.2f} s  {evaluations:6d} evals  "
            f"{result.nit:4d} iters  merit {float(result.fun):.6g} "
            f"(Optiland re-score {rescored:.6g})",
            flush=True,
        )
    return out


# --------------------------------------------------------------------------


def environment() -> dict:
    """Hardware and versions recorded with the results."""
    cpu = "unknown"
    with contextlib.suppress(OSError, StopIteration), open("/proc/cpuinfo") as f:
        cpu = next(
            ln.split(":", 1)[1].strip() for ln in f if ln.startswith("model name")
        )
    rustc = subprocess.run(
        ["rustc", "--version"], capture_output=True, text=True, check=False
    ).stdout.strip()
    return {
        "cpu": cpu,
        "threads": THREADS,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "numba": numba.__version__,
        "rustc": rustc,
        "date": time.strftime("%Y-%m-%d"),
    }


def main():
    """Run the selected sections and write the JSON."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default=",".join(SECTIONS))
    parser.add_argument("--out", default="small_ops.json", help="file in results/")
    parser.add_argument(
        "--skip-slow", action="store_true", help="skip the multi-minute Python runs"
    )
    args = parser.parse_args()

    w = Workload()
    results = {"environment": environment()}
    print("Environment:", results["environment"])
    runners = {
        "boundary": bench_boundary,
        "leaf": bench_leaf,
        "patches": bench_patches,
        "compiled": bench_compiled,
        "gradient": bench_gradient,
        "optimize": lambda wl: bench_optimize(wl, args.skip_slow),
        "blas": bench_blas,
    }
    for section in args.only.split(","):
        print(f"\n== {section} ==")
        results[section] = runners[section](w)

    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / args.out
    if path.exists():  # keep sections from earlier partial runs
        previous = json.loads(path.read_text())
        previous.update(results)
        results = previous
    path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
