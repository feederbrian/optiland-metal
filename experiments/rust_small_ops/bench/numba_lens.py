"""Compiled-lens merit in Numba: the same algorithm as ``src/lens.rs``.

``NumbaLens`` packs the snapshot into arrays once; ``merit_at(x)`` is then a
single jitted call, like ``CompiledLens.merit_at`` on the Rust side.
"""

from __future__ import annotations

import math

import numpy as np
from lean import hexapolar
from numba import njit, prange
from snapshot import PLANE, Snapshot

EPS = np.finfo(np.float64).eps
RADIUS, THICKNESS = 0, 1

_JIT = {"cache": True, "error_model": "numpy"}


@njit(inline="always", **_JIT)
def _conic_distance(x, y, z, L, M, N, radius, conic):
    k1 = 1.0 + conic
    kz = k1 * z
    transverse = x * x + y * y
    transverse_dir = L * L + M * M
    N2 = N * N
    a = transverse_dir + k1 * N2
    b = 2.0 * (L * x + M * y + N * (kz - radius))
    c = transverse + z * (kz - 2.0 * radius)
    residual_scale = transverse + abs(z) * (abs(kz) + 2.0 * abs(radius))
    roundoff = 4.0 * EPS
    resolved_c = abs(c) > roundoff * residual_scale
    d = b * b - 4.0 * a * c
    sqrt_d = math.sqrt(d) if d > 0.0 else 0.0
    q = -0.5 * (b + math.copysign(sqrt_d, b))
    a_ok = a != 0.0 and abs(a) < math.inf
    q_ok = q != 0.0 and abs(q) < math.inf
    t1 = q / (a if a_ok else 1.0)
    t2 = c / (q if q_ok else 1.0)
    resolved_step = t2 * t2 * (transverse_dir + N2) > roundoff * roundoff * (
        transverse + z * z
    )
    solvable1 = d >= 0.0 and a_ok and abs(t1) < math.inf
    solvable2 = d >= 0.0 and q_ok and abs(t2) < math.inf
    z1 = z + t1 * N
    z2 = z + t2 * N
    valid1 = solvable1 and t1 > 0.0 and 1.0 - k1 * z1 / radius >= 0.0
    valid2 = (
        solvable2
        and (resolved_c or resolved_step)
        and t2 > 0.0
        and 1.0 - k1 * z2 / radius >= 0.0
    )
    v2 = abs(z2) if solvable2 else math.inf
    v1 = abs(z1) if solvable1 else math.inf
    pick2 = valid2 or (not valid1 and v2 < v1)
    if solvable1 or solvable2:
        return t2 if pick2 else t1
    return math.nan


@njit(**_JIT)
def _positions(thickness, first_z):
    z = np.full(thickness.shape[0], -np.inf)
    z[1] = first_z
    for k in range(1, z.shape[0] - 1):
        z[k + 1] = z[k] + thickness[k]
    return z


@njit(**_JIT)
def _paraxial(kinds, radius, thickness, n, y, u, last, refract_last):
    for k in range(1, last + 1):
        if k > 1:
            y += u * thickness[k - 1]
        if k == last and not refract_last:
            break
        c = 0.0 if kinds[k] == PLANE else 1.0 / radius[k]
        u = (n[k - 1] * u - y * (n[k] - n[k - 1]) * c) / n[k]
    return y, u


@njit(**_JIT)
def _f2(kinds, radius, thickness, n):
    _, u = _paraxial(kinds, radius, thickness, n, 1.0, 0.0, kinds.shape[0] - 1, True)
    return -1.0 / u


@njit(**_JIT)
def _epl(kinds, radius, thickness, n, stop):
    if stop == 1:
        return 0.0
    a, _ = _paraxial(kinds, radius, thickness, n, 1.0, 0.0, stop, False)
    b, _ = _paraxial(kinds, radius, thickness, n, 0.0, 1.0, stop, False)
    return b / a


@njit(**_JIT)
def _trace(kinds, radius, conic, z, n, x, y, zz, L, M, N):
    for k in range(1, kinds.shape[0]):
        zz -= z[k]
        R = radius[k]
        kc = conic[k]
        is_plane = kinds[k] == PLANE
        t = -zz / N if is_plane else _conic_distance(x, y, zz, L, M, N, R, kc)
        x += t * L
        y += t * M
        zz += t * N
        if kinds[k] == PLANE:
            nx, ny, nz = 0.0, 0.0, 1.0
        else:
            denom = R * math.sqrt(1.0 - (1.0 + kc) * (x * x + y * y) / (R * R))
            fx = x / denom
            fy = y / denom
            mag = math.sqrt(fx * fx + fy * fy + 1.0)
            nx, ny, nz = fx / mag, fy / mag, -1.0 / mag
        dot = L * nx + M * ny + N * nz
        sgn = np.sign(dot)
        nx, ny, nz, dot = nx * sgn, ny * sgn, nz * sgn, abs(dot)
        u = n[k - 1] / n[k]
        root = math.sqrt(1.0 - u * u * (1.0 - dot * dot))
        L, M, N = (
            u * L + nx * root - u * nx * dot,
            u * M + ny * root - u * ny * dot,
            u * N + nz * root - u * nz * dot,
        )
        zz += z[k]
    return x, y, zz, L, M, N


@njit(**_JIT)
def _rms_spot(
    kinds, radius, conic, z, index, epd, max_field, primary, epl, px, py, hx, hy
):
    fx = max_field * hx
    fy = max_field * hy
    min_rel = np.inf
    for k in range(1, z.shape[0] - 1):
        min_rel = min(min_rel, z[k] - z[1])
    d = (epd - min_rel) + epl
    sign = -1.0 if math.cos(math.radians(math.sqrt(fx * fx + fy * fy))) < 0 else 1.0
    z_pupil = epl + z[1]
    x_off = -sign * math.tan(math.radians(fx)) * d
    y_off = -sign * math.tan(math.radians(fy)) * d
    z0 = z_pupil - sign * d
    waves, npts = index.shape[0], px.shape[0]
    xs = np.empty(waves * npts)
    ys = np.empty(waves * npts)
    for w in range(waves):
        for j in range(npts):
            x1 = px[j] * epd / 2.0
            y1 = py[j] * epd / 2.0
            x0 = x1 + x_off
            y0 = y1 + y_off
            dx, dy, dz = x1 - x0, y1 - y0, z_pupil - z0
            mag = math.sqrt(dx * dx + dy * dy + dz * dz)
            if mag < 1e-9:
                L, M, N = 0.0, 0.0, 1.0
            else:
                L, M, N = dx / mag, dy / mag, dz / mag
            r = _trace(kinds, radius, conic, z, index[w], x0, y0, z0, L, M, N)
            xs[w * npts + j] = r[0]
            ys[w * npts + j] = r[1]
    if np.isnan(xs).any() or np.isnan(ys).any():
        return np.nan
    p = primary * npts
    cx = xs[p : p + npts].mean()
    cy = ys[p : p + npts].mean()
    return math.sqrt(np.mean((xs - cx) ** 2 + (ys - cy) ** 2))


@njit(**_JIT)
def merit_at(
    x, kinds, radius0, conic, thickness0, index, first_z, stop, epd, max_field,
    primary, px, py, f2_weight, f2_target, spots, var_kind, var_surface,
    var_factor, var_offset,
):  # fmt: skip
    """Merit at scaled variable vector ``x`` (the snapshot is not modified)."""
    radius = radius0.copy()
    thickness = thickness0.copy()
    for i in range(x.shape[0]):
        value = (x[i] - var_offset[i]) / var_factor[i]
        if var_kind[i] == RADIUS:
            radius[var_surface[i]] = value
        else:
            thickness[var_surface[i]] = value
    z = _positions(thickness, first_z)
    n = index[primary]
    term = f2_weight * (_f2(kinds, radius, thickness, n) - f2_target)
    total = term * term
    epl = _epl(kinds, radius, thickness, n, stop)
    for s in range(spots.shape[0]):
        rms = _rms_spot(
            kinds, radius, conic, z, index, epd, max_field, primary, epl, px, py,
            spots[s, 0], spots[s, 1],
        )  # fmt: skip
        term = spots[s, 2] * (rms - spots[s, 3])
        total += term * term
    return total


@njit(**_JIT)
def rms_spot_one(
    kinds, radius, conic, thickness, index, first_z, stop, epd, max_field, primary,
    px, py, hx, hy,
):  # fmt: skip
    """One multi-wavelength RMS spot, paraxials included, in one call."""
    z = _positions(thickness, first_z)
    epl = _epl(kinds, radius, thickness, index[primary], stop)
    return _rms_spot(
        kinds, radius, conic, z, index, epd, max_field, primary, epl, px, py, hx, hy
    )


@njit(**_JIT)
def trace_one(
    kinds, radius, conic, thickness, index, first_z, stop, epd, max_field, primary,
    hx, hy, px, py, w,
):  # fmt: skip
    """``trace_generic`` for one ray: paraxials, aiming and trace in one call."""
    z = _positions(thickness, first_z)
    epl = _epl(kinds, radius, thickness, index[primary], stop)
    fx, fy = max_field * hx, max_field * hy
    min_rel = np.inf
    for k in range(1, z.shape[0] - 1):
        min_rel = min(min_rel, z[k] - z[1])
    d = (epd - min_rel) + epl
    sign = -1.0 if math.cos(math.radians(math.sqrt(fx * fx + fy * fy))) < 0 else 1.0
    z_pupil = epl + z[1]
    x1, y1 = px * epd / 2.0, py * epd / 2.0
    x0 = x1 - sign * math.tan(math.radians(fx)) * d
    y0 = y1 - sign * math.tan(math.radians(fy)) * d
    z0 = z_pupil - sign * d
    dx, dy, dz = x1 - x0, y1 - y0, z_pupil - z0
    mag = math.sqrt(dx * dx + dy * dy + dz * dz)
    return _trace(
        kinds, radius, conic, z, index[w], x0, y0, z0, dx / mag, dy / mag, dz / mag
    )


@njit(parallel=True, **_JIT)
def merit_many(xs, *lens):
    """``merit_at`` for every row of ``xs``, rows in parallel."""
    out = np.empty(xs.shape[0])
    for i in prange(xs.shape[0]):
        out[i] = merit_at(xs[i], *lens)
    return out


class NumbaLens:
    """Arrays packed once from a snapshot; each merit is one jitted call."""

    def __init__(self, snap: Snapshot):
        pupil = hexapolar(snap.num_rings)
        self.args = (
            np.asarray(snap.kinds, np.int64),
            np.asarray(snap.radius, np.float64),
            np.asarray(snap.conic, np.float64),
            np.asarray(snap.thickness, np.float64),
            np.asarray(snap.index, np.float64),
            float(snap.first_z),
            int(snap.stop),
            float(snap.epd),
            float(snap.max_field),
            int(snap.primary),
            np.array([p[0] for p in pupil]),
            np.array([p[1] for p in pupil]),
            float(snap.f2_weight),
            float(snap.f2_target),
            np.asarray(snap.spots, np.float64).reshape(-1, 4),
            np.array(
                [RADIUS if v[0] == "radius" else THICKNESS for v in snap.variables]
            ),
            np.array([v[1] for v in snap.variables], np.int64),
            np.array([v[2] for v in snap.variables], np.float64),
            np.array([v[3] for v in snap.variables], np.float64),
        )

    def merit_at(self, x) -> float:
        return merit_at(np.asarray(x, np.float64), *self.args)

    def f2(self) -> float:
        kinds, radius, _, thickness, index = self.args[:5]
        return _f2(kinds, radius, thickness, index[self.args[9]])

    def rms_spot(self, hx, hy) -> float:
        return rms_spot_one(*self.args[:12], float(hx), float(hy))

    def trace_generic(self, hx, hy, px, py, w):
        return trace_one(*self.args[:10], hx, hy, px, py, w)

    def merit_many(self, xs) -> np.ndarray:
        return merit_many(np.ascontiguousarray(xs, np.float64), *self.args)
