"""Count the ray-surface intersections in one merit evaluation after a step.

Splits them into the aimer's internal traces and the operands' own traces,
and counts the even-asphere Newton residual evaluations. Multiplied by the
per-intersection cost from ``compiled_floor.py``, this is the arithmetic
floor of a compiled merit using the same algorithm.

    .venv/bin/python experiments/uw2_workload/bench/count_work.py
"""

from __future__ import annotations

import argparse
import collections
import warnings

import numpy as np
import uw1_optic
import uw2_operands as ops

from optiland.geometries.newton_raphson import NewtonRaphsonGeometry
from optiland.rays.ray_aiming.iterative import IterativeRayAimer
from optiland.surfaces import standard_surface

warnings.filterwarnings("ignore")


def main():
    """Count intersections during one perturbed proxy merit."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--optic", default=None)
    args = parser.parse_args()

    lens = uw1_optic.load(args.optic)
    ops.evaluate(lens)
    r0 = float(lens.surfaces.surfaces[3].geometry.radius)
    lens.updater.set_radius(r0 * (1 + 1e-5), 3)
    ops.evaluate(lens)  # a first step: the aimer has history, as in a loop

    phase = ["operands"]
    counts: collections.Counter = collections.Counter()
    coordinator = standard_surface._TracingCoordinator
    trace, subset = coordinator.trace, IterativeRayAimer._trace_subset
    residual = NewtonRaphsonGeometry._surface_residual

    def counted_trace(self, rays, surface, record=True):
        asphere = isinstance(surface.geometry, NewtonRaphsonGeometry)
        counts[phase[0], "asphere" if asphere else "other"] += int(np.size(rays.x))
        counts[phase[0], "surface calls"] += 1
        return trace(self, rays, surface, record)

    def counted_subset(self, *a, **k):
        previous, phase[0] = phase[0], "aimer"
        try:
            return subset(self, *a, **k)
        finally:
            phase[0] = previous

    def counted_residual(self, t, rays):
        counts[phase[0], "asphere residuals"] += int(np.size(rays.x))
        return residual(self, t, rays)

    coordinator.trace = counted_trace
    IterativeRayAimer._trace_subset = counted_subset
    NewtonRaphsonGeometry._surface_residual = counted_residual
    try:
        lens.updater.set_radius(r0 * (1 + 2e-5), 3)
        ops.evaluate(lens)
    finally:
        coordinator.trace = trace
        IterativeRayAimer._trace_subset = subset
        NewtonRaphsonGeometry._surface_residual = residual

    for (where, what), value in sorted(counts.items()):
        print(f"{where:9s} {what:18s} {value:>12,}")
    total = sum(v for (_, what), v in counts.items() if what in ("asphere", "other"))
    print(f"ray-surface intersections per merit: {total:,}")


if __name__ == "__main__":
    main()
