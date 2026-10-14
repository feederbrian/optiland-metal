"""Measure what Optiland's small operations cost, and where the time goes.

    .venv/bin/python experiments/rust_small_ops/bench/characterize.py

Writes ``results/characterize.json`` (or ``--out NAME``).
"""

from __future__ import annotations

import cProfile
import json
import pstats
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import workload as common

from optiland import optimization
from optiland.samples.objectives import CookeTriplet

RESULTS = Path(__file__).resolve().parents[1] / "results"

# Self time is attributed to the first matching path fragment.
CATEGORIES = (
    ("optiland/backend/", "backend dispatch (be.*)"),
    ("optiland/materials/", "material index + cache keys"),
    ("optiland/paraxial", "paraxial path + paraxial traces"),
    ("optiland/raytrace/paraxial", "paraxial path + paraxial traces"),
    ("optiland/rays/ray_aiming", "ray generation + aiming"),
    ("optiland/rays/ray_generator", "ray generation + aiming"),
    ("optiland/fields/", "ray generation + aiming"),
    ("optiland/distribution", "ray generation + aiming"),
    ("optiland/coordinate_system", "coordinate frames"),
    ("optiland/rays/", "real-ray physics (Python side)"),
    ("optiland/surfaces/", "real-ray physics (Python side)"),
    ("optiland/geometries/", "real-ray physics (Python side)"),
    ("optiland/interactions/", "real-ray physics (Python side)"),
    ("optiland/propagation/", "real-ray physics (Python side)"),
    ("optiland/raytrace/", "real-ray physics (Python side)"),
    ("optiland/optimization/", "optimization bookkeeping"),
    ("optiland/", "other Optiland Python"),
    ("numpy", "NumPy (Python + C)"),
)


def categorize(stats: pstats.Stats) -> dict[str, float]:
    """Share of total self time per category."""
    buckets: dict[str, float] = defaultdict(float)
    total = 0.0
    for (filename, _, name), (_, _, tottime, _, _) in stats.stats.items():
        total += tottime
        label = next((lab for frag, lab in CATEGORIES if frag in filename), None)
        if label is None:
            is_numpy = "numpy" in name or "ufunc" in name or "ndarray" in name
            label = "NumPy (Python + C)" if is_numpy else "Python builtins + other"
        buckets[label] += tottime
    return {k: v / total for k, v in sorted(buckets.items(), key=lambda kv: -kv[1])}


def profile(fn, repeats=20) -> dict:
    """Function calls per invocation and the self-time split."""
    fn()
    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(repeats):
        fn()
    profiler.disable()
    stats = pstats.Stats(profiler)
    return {
        "calls_per_invocation": stats.total_calls / repeats,
        "shares": categorize(stats),
    }


def latency_table(lens, wl) -> dict[str, float]:
    """Per-call latency of individual small operations on an unchanged lens."""
    glass = lens.surfaces[1].material_post
    ops = {
        "paraxial.EPD()": lambda: lens.paraxial.EPD(),
        "paraxial.f2()": lambda: lens.paraxial.f2(),
        "paraxial.EPL()": lambda: lens.paraxial.EPL(),
        "paraxial.XPL()": lambda: lens.paraxial.XPL(),
        "material.n(wl)": lambda: glass.n(wl),
        "trace_generic, 1 ray": lambda: lens.trace_generic(0.0, 1.0, 0.0, 1.0, wl),
        "trace, 91 rays (5-ring hexapolar)": lambda: lens.trace(
            0, 1, wl, common.NUM_RINGS, "hexapolar"
        ),
        "updater.update()": lambda: lens.updater.update(),
    }
    out = {}
    for name, fn in ops.items():
        out[name] = common.per_call(fn)
        print(f"  {name:36s} {common.fmt(out[name]):>12s}", flush=True)
    return out


def ray_scaling(lens, wl) -> list[dict]:
    """Trace latency vs ray count: fixed overhead vs per-ray cost."""
    rows = []
    for rings in (0, 1, 3, 10, 30, 100, 300):
        n = lens.trace(0, 1, wl, rings, "hexapolar").x.size
        t = common.per_call(
            lambda r=rings: lens.trace(0, 1, wl, r, "hexapolar", record=False),
            repeats=3,
        )
        rows.append({"rays": int(n), "seconds": t})
        print(f"  {n:8,d} rays  {common.fmt(t):>12s}  {t / n * 1e9:10.0f} ns/ray")
    return rows


def merit_costs(problem) -> dict[str, float]:
    """Merit evaluation the way the optimizer does it (variables perturbed)."""
    x0 = np.array([float(v.value) for v in problem.variables])
    optimizer = optimization.OptimizerGeneric(problem)
    state = {"k": 0}

    def perturbed():
        # Alternate a tiny step so caches cannot serve a stale result.
        state["k"] ^= 1
        optimizer._fun(x0 * (1 + 1e-9 * state["k"]))

    out = {
        "problem.sum_squared() (unchanged lens)": common.per_call(problem.sum_squared),
        "optimizer._fun(x) (update + merit)": common.per_call(perturbed),
    }
    for name, t in out.items():
        print(f"  {name:44s} {common.fmt(t):>12s}")
    return out


def optimize_run() -> dict:
    """The tutorial's OptimizerGeneric run, counting merit evaluations."""
    lens = common.tutorial_triplet()
    problem = common.tutorial_problem(lens)
    initial = float(problem.sum_squared())
    optimizer = optimization.OptimizerGeneric(problem)
    calls = {"n": 0}
    original = optimizer._fun

    def counted(x):
        calls["n"] += 1
        return original(x)

    optimizer._fun = counted
    start = time.perf_counter()
    result = optimizer.optimize(tol=1e-9, disp=False)
    wall = time.perf_counter() - start
    out = {
        "wall_seconds": wall,
        "merit_evaluations": calls["n"],
        "iterations": int(result.nit),
        "seconds_per_evaluation": wall / calls["n"],
        "initial_merit": initial,
        "final_merit": float(result.fun),
    }
    print(
        f"  {out['iterations']} iterations, {out['merit_evaluations']} merit "
        f"evaluations, {wall:.1f} s total, "
        f"{common.fmt(out['seconds_per_evaluation'])} per evaluation, "
        f"final merit {out['final_merit']:.4g}"
    )
    return out


def main():
    """Run every characterization and write the JSON."""
    import argparse  # noqa: PLC0415

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="characterize.json", help="file in results/")
    args = parser.parse_args()
    results = {}
    lens = CookeTriplet()
    wl = lens.primary_wavelength
    print("Per-call latency, Cooke triplet sample (7 surfaces):")
    results["latency"] = latency_table(lens, wl)
    print("\nTrace latency vs ray count:")
    results["ray_scaling"] = ray_scaling(lens, wl)

    print("\nTutorial merit function (f2 + 3 fields x 3 wavelengths RMS spot):")
    problem = common.tutorial_problem(common.tutorial_triplet())
    results["merit"] = merit_costs(problem)

    print("\nWhere a single-ray trace and a merit evaluation spend time:")
    results["profile"] = {
        "trace_generic, 1 ray": profile(
            lambda: lens.trace_generic(0.0, 1.0, 0.0, 1.0, wl)
        ),
        "problem.sum_squared()": profile(problem.sum_squared, repeats=5),
    }
    for name, prof in results["profile"].items():
        print(f"  {name}: {prof['calls_per_invocation']:,.0f} Python calls")
        for label, share in prof["shares"].items():
            if share >= 0.01:
                print(f"      {share:5.0%}  {label}")

    print("\nFull tutorial optimization (OptimizerGeneric, tol=1e-9):")
    results["optimize"] = optimize_run()

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / args.out).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
