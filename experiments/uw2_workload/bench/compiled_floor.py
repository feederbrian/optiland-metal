"""What does one ray-surface intersection cost on this lens in compiled code?

A Numba kernel traces launch rays through the UW2 lens (planes, spheres and
conics in closed form; even aspheres by the same conic-seeded Newton
iteration Optiland uses), checks the image-plane rays against Optiland's own
trace of the same launch, and times bundles large enough that call overhead
vanishes. Together with the intersection count of one merit evaluation this
gives the floor of a compiled merit: the arithmetic alone.

    .venv/bin/python experiments/uw2_workload/bench/compiled_floor.py
"""

from __future__ import annotations

import argparse
import math
import time
import warnings

import numpy as np
import uw1_optic
from numba import get_num_threads, njit, prange

import optiland.backend as be
from optiland.geometries import Plane
from optiland.geometries.newton_raphson import NewtonRaphsonGeometry

warnings.filterwarnings("ignore")

PLANE, CONIC, ASPHERE = 0, 1, 2
MAX_TERMS = 12
_JIT = {"cache": True, "error_model": "numpy", "fastmath": False}


def snapshot(lens, wavelength: float) -> tuple:
    """Surfaces 1..last as flat arrays at one wavelength."""
    surfaces = lens.surfaces.surfaces[1:]
    count = len(surfaces)
    kind = np.zeros(count, np.int64)
    radius = np.full(count, np.inf)
    conic = np.zeros(count)
    coeffs = np.zeros((count, MAX_TERMS))
    tol = np.zeros(count)
    max_iter = np.zeros(count, np.int64)
    z = np.ravel(be.to_numpy(lens.surfaces.positions))[1:].astype(float)
    n_before = np.zeros(count)
    n_after = np.zeros(count)
    w = np.array([wavelength])
    for i, s in enumerate(surfaces):
        g = s.geometry
        if any(float(getattr(g.cs, a)) for a in ("x", "y", "rx", "ry", "rz")):
            raise NotImplementedError("decentered or tilted surface")
        n_before[i] = float(np.ravel(s.material_pre.n(w))[0])
        n_after[i] = float(np.ravel(s.material_post.n(w))[0])
        if isinstance(g, Plane):
            continue
        radius[i] = float(g.radius)
        conic[i] = float(g.k)
        kind[i] = CONIC
        if isinstance(g, NewtonRaphsonGeometry):
            kind[i] = ASPHERE
            c = [float(v) for v in g.coefficients]
            coeffs[i, : len(c)] = c
            tol[i] = float(g.tol)
            max_iter[i] = int(g.max_iter)
    return kind, radius, conic, coeffs, tol, max_iter, z, n_before, n_after


@njit(inline="always", **_JIT)
def _conic_t(x, y, z, L, M, N, R, k):
    # Closed-form intersection with z = c r^2 / (1 + sqrt(1 - (1+k) c^2 r^2)),
    # root nearest the vertex plane (the physical branch for these rays).
    k1 = 1.0 + k
    a = L * L + M * M + k1 * N * N
    b = 2.0 * (L * x + M * y + N * (k1 * z - R))
    c = x * x + y * y + z * (k1 * z - 2.0 * R)
    d = b * b - 4.0 * a * c
    if d < 0.0:
        return math.nan
    q = -0.5 * (b + math.copysign(math.sqrt(d), b))
    t1 = q / a if a != 0.0 else math.inf
    t2 = c / q if q != 0.0 else math.inf
    z1 = abs(z + t1 * N)
    z2 = abs(z + t2 * N)
    return t2 if z2 < z1 else t1


@njit(inline="always", **_JIT)
def _sag(r2, R, k, coeffs):
    s = r2 / (R * (1.0 + math.sqrt(1.0 - (1.0 + k) * r2 / (R * R))))
    p = r2
    for c in coeffs:
        s += c * p
        p *= r2
    return s


@njit(inline="always", **_JIT)
def _slopes(x, y, R, k, coeffs):
    r2 = x * x + y * y
    denom = R * math.sqrt(1.0 - (1.0 + k) * r2 / (R * R))
    fx = x / denom
    fy = y / denom
    p = 1.0
    for i in range(coeffs.shape[0]):
        g = 2.0 * (i + 1) * coeffs[i] * p
        fx += g * x
        fy += g * y
        p *= r2
    return fx, fy


@njit(**_JIT)
def _trace_ray(snap, x, y, zz, L, M, N):
    kind, radius, conic, coeffs, tol, max_iter, z, n_before, n_after = snap
    newton = 0
    for s in range(kind.shape[0]):
        zl = zz - z[s]
        if kind[s] == PLANE:
            t = -zl / N
        else:
            R, k = radius[s], conic[s]
            t = _conic_t(x, y, zl, L, M, N, R, k)
            if kind[s] == ASPHERE:
                for _ in range(max_iter[s]):
                    xi, yi = x + t * L, y + t * M
                    f = _sag(xi * xi + yi * yi, R, k, coeffs[s]) - (zl + t * N)
                    if not abs(f) >= tol[s]:  # converged, or NaN
                        break
                    fx, fy = _slopes(xi, yi, R, k, coeffs[s])
                    t -= f / (fx * L + fy * M - N)
                    newton += 1
        x += t * L
        y += t * M
        zl += t * N
        if kind[s] == PLANE:
            nx, ny, nz = 0.0, 0.0, -1.0
        else:
            fx, fy = _slopes(x, y, radius[s], conic[s], coeffs[s])
            mag = math.sqrt(fx * fx + fy * fy + 1.0)
            nx, ny, nz = fx / mag, fy / mag, -1.0 / mag
        cos_i = -(L * nx + M * ny + N * nz)
        if cos_i < 0.0:
            nx, ny, nz, cos_i = -nx, -ny, -nz, -cos_i
        mu = n_before[s] / n_after[s]
        root = 1.0 - mu * mu * (1.0 - cos_i * cos_i)
        root = math.sqrt(root) if root >= 0.0 else math.nan
        g = mu * cos_i - root
        L, M, N = mu * L + g * nx, mu * M + g * ny, mu * N + g * nz
        zz = zl + z[s]
    return x, y, zz, L, M, N, newton


@njit(parallel=True, **_JIT)
def trace_bundle(snap, x, y, z, L, M, N):
    """Trace each launch ray through every surface; (x, y, z, L, M, N, steps)."""
    count = x.shape[0]
    out = np.empty((6, count))
    steps = np.zeros(count, np.int64)
    for i in prange(count):
        xo, yo, zo, Lo, Mo, No, steps[i] = _trace_ray(
            snap, x[i], y[i], z[i], L[i], M[i], N[i]
        )
        out[0, i], out[1, i], out[2, i] = xo, yo, zo
        out[3, i], out[4, i], out[5, i] = Lo, Mo, No
    return out, steps


def main():
    """Validate the kernel against Optiland, then time it."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--optic", default=None)
    parser.add_argument("--rays", type=int, default=200_000)
    args = parser.parse_args()

    lens = uw1_optic.load(args.optic)
    wl = 0.5876
    snap = snapshot(lens, wl)
    count = len(snap[0])
    worst = 0.0
    for u in (0.0, 0.7, 1.0):
        px, py = (np.ravel(be.to_numpy(a)) for a in _hexapolar(6))
        launch = lens.ray_tracer.ray_generator.generate_rays(
            np.zeros(px.size), np.full(px.size, u), px, py, wl
        )
        start = [np.array(np.ravel(be.to_numpy(getattr(launch, f)))) for f in "xyzLMN"]
        lens.surfaces.trace(launch, skip=1, record=False)
        ref = np.array([np.ravel(be.to_numpy(getattr(launch, f))) for f in "xyzLMN"])
        out, _ = trace_bundle(snap, *start)
        ok = np.isfinite(ref).all(axis=0) & np.isfinite(out).all(axis=0)
        worst = max(worst, float(np.max(np.abs(out[:, ok] - ref[:, ok]))))
    print(f"kernel vs Optiland, rays at the image surface: max |diff| {worst:.2e}")

    rng = np.random.default_rng(0)
    pick = rng.integers(0, start[0].shape[0], args.rays)
    bundle = [np.ascontiguousarray(a[pick]) for a in start]
    trace_bundle(snap, *bundle)
    for threads in (1, get_num_threads()):
        _set_threads(threads)
        best = math.inf
        for _ in range(5):
            t0 = time.perf_counter()
            _, steps = trace_bundle(snap, *bundle)
            best = min(best, time.perf_counter() - t0)
        per = best / (args.rays * count) * 1e9
        print(
            f"{threads} thread(s): {per:.1f} ns per ray-surface "
            f"({count} surfaces, {steps.mean():.2f} Newton steps per ray)"
        )


def _hexapolar(rings: int):
    from optiland.distribution import create_distribution

    d = create_distribution("hexapolar")
    d.generate_points(rings)
    return d.x, d.y


def _set_threads(n: int) -> None:
    from numba import set_num_threads

    set_num_threads(n)


if __name__ == "__main__":
    main()
