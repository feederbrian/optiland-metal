"""Compiled-lens merit in plain Python and in lean NumPy (control group).

Both run the identical algorithm to ``src/lens.rs``, but in the interpreter:

* ``PythonLens``: scalar ``math`` code, one ray at a time. Its distance to
  Rust is the cost of the interpreter itself, with Optiland's object model
  removed.
* ``NumpyLens``: rays vectorized with NumPy, no Optiland object model. Its
  distance to Optiland is the cost of the object model; its distance to
  Rust is NumPy's per-call overhead on small arrays.
"""

from __future__ import annotations

import copy
import math

import numpy as np
from snapshot import CONIC, PLANE, Snapshot

EPS = np.finfo(np.float64).eps


def hexapolar(num_rings: int) -> list[tuple[float, float]]:
    """``HexagonalDistribution.generate_points``."""
    points = [(0.0, 0.0)]
    for i in range(num_rings):
        r = (i + 1) / num_rings
        count = 6 * (i + 1)
        for j in range(count):
            theta = 2.0 * math.pi * j / count
            points.append((r * math.cos(theta), r * math.sin(theta)))
    return points


class _Base:
    """State shared by both interpreters: variables, positions, paraxials."""

    def __init__(self, snap: Snapshot):
        self.s = copy.deepcopy(snap)
        self.pupil = hexapolar(snap.num_rings)
        self.refresh_positions()

    def refresh_positions(self):
        s = self.s
        z = [-math.inf] * len(s.kinds)
        z[1] = s.first_z
        for k in range(1, len(s.kinds) - 1):
            z[k + 1] = z[k] + s.thickness[k]
        self.z = z

    def apply(self, x):
        for (kind, surface, factor, offset), xi in zip(
            self.s.variables, x, strict=True
        ):
            value = (float(xi) - offset) / factor
            if kind == "radius":
                self.s.radius[surface] = value
            else:
                self.s.thickness[surface] = value
        self.refresh_positions()

    def paraxial(self, w, y, u, last, refract_last):
        s = self.s
        n = s.index[w]
        for k in range(1, last + 1):
            if k > 1:
                y += u * s.thickness[k - 1]
            if k == last and not refract_last:
                break
            c = 0.0 if s.kinds[k] == PLANE else 1.0 / s.radius[k]
            u = (n[k - 1] * u - y * (n[k] - n[k - 1]) * c) / n[k]
        return y, u

    def f2(self):
        _, u = self.paraxial(self.s.primary, 1.0, 0.0, len(self.s.kinds) - 1, True)
        return -1.0 / u

    def epl(self):
        s = self.s
        if s.stop == 1:
            return 0.0
        a, _ = self.paraxial(s.primary, 1.0, 0.0, s.stop, False)
        b, _ = self.paraxial(s.primary, 0.0, 1.0, s.stop, False)
        return b / a

    def launch_geometry(self, epl, hx, hy):
        """Per-field launch constants: (x, y, z offsets, z_pupil)."""
        s, z = self.s, self.z
        fx, fy = s.max_field * hx, s.max_field * hy
        min_rel = min(z[k] - z[1] for k in range(1, len(z) - 1))
        d = (s.epd - min_rel) + epl
        sign = -1.0 if math.cos(math.radians(math.hypot(fx, fy))) < 0 else 1.0
        z_pupil = epl + z[1]
        x = -sign * math.tan(math.radians(fx)) * d
        y = -sign * math.tan(math.radians(fy)) * d
        return x, y, z_pupil - sign * d, z_pupil

    def merit(self):
        s = self.s
        term = s.f2_weight * (self.f2() - s.f2_target)
        total = term * term
        epl = self.epl()
        for hx, hy, weight, target in s.spots:
            term = weight * (self.rms_spot(epl, hx, hy) - target)
            total += term * term
        return total

    def merit_at(self, x):
        work = copy.copy(self)
        work.s = copy.copy(self.s)
        work.s.radius = list(self.s.radius)
        work.s.thickness = list(self.s.thickness)
        work.apply(x)
        return work.merit()


class PythonLens(_Base):
    """Scalar plain-Python implementation."""

    def launch(self, epl, hx, hy, px, py, geometry=None):
        x, y, z0, z_pupil = geometry or self.launch_geometry(epl, hx, hy)
        x1, y1 = px * self.s.epd / 2.0, py * self.s.epd / 2.0
        x0, y0 = x1 + x, y1 + y
        dx, dy, dz = x1 - x0, y1 - y0, z_pupil - z0
        mag = math.sqrt(dx * dx + dy * dy + dz * dz)
        if mag < 1e-9:
            return [x0, y0, z0, 0.0, 0.0, 1.0]
        return [x0, y0, z0, dx / mag, dy / mag, dz / mag]

    def trace(self, w, ray):
        s, z = self.s, self.z
        n = s.index[w]
        x, y, zz, L, M, N = ray
        for k in range(1, len(s.kinds)):
            zz -= z[k]
            R, kc = s.radius[k], s.conic[k]
            if s.kinds[k] == PLANE:
                t = -zz / N if N != 0.0 else math.copysign(math.inf, -zz)
            else:
                t = _conic_distance(x, y, zz, L, M, N, R, kc)
            x, y, zz = x + t * L, y + t * M, zz + t * N
            if s.kinds[k] == PLANE:
                nx, ny, nz = 0.0, 0.0, 1.0
            else:
                r2 = x * x + y * y
                arg = 1.0 - (1.0 + kc) * r2 / (R * R)
                denom = R * math.sqrt(arg) if arg >= 0 else math.nan
                fx, fy = x / denom, y / denom
                mag = math.sqrt(fx * fx + fy * fy + 1.0)
                nx, ny, nz = fx / mag, fy / mag, -1.0 / mag
            dot = L * nx + M * ny + N * nz
            sgn = (dot > 0) - (dot < 0) if dot == dot else math.nan
            nx, ny, nz, dot = nx * sgn, ny * sgn, nz * sgn, abs(dot)
            u = n[k - 1] / n[k]
            arg = 1.0 - u * u * (1.0 - dot * dot)
            root = math.sqrt(arg) if arg >= 0 else math.nan
            L, M, N = (
                u * L + nx * root - u * nx * dot,
                u * M + ny * root - u * ny * dot,
                u * N + nz * root - u * nz * dot,
            )
            zz += z[k]
        return [x, y, zz, L, M, N]

    def trace_generic(self, hx, hy, px, py, w):
        epl = self.epl()
        return self.trace(w, self.launch(epl, hx, hy, px, py))

    def rms_spot(self, epl, hx, hy):
        geometry = self.launch_geometry(epl, hx, hy)
        xs, ys = [], []
        for w in range(len(self.s.index)):
            for px, py in self.pupil:
                r = self.trace(w, self.launch(epl, hx, hy, px, py, geometry))
                xs.append(r[0])
                ys.append(r[1])
        if any(v != v for v in xs + ys):
            return math.nan
        np_ = len(self.pupil)
        p = self.s.primary * np_
        cx = sum(xs[p : p + np_]) / np_
        cy = sum(ys[p : p + np_]) / np_
        r2 = sum((x - cx) ** 2 + (y - cy) ** 2 for x, y in zip(xs, ys, strict=True))
        return math.sqrt(r2 / len(xs))


def _conic_distance(x, y, z, L, M, N, radius, conic):
    """``optiland.backend._conic`` root selection, scalar."""
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
    z1, z2 = z + t1 * N, z + t2 * N
    valid1 = solvable1 and t1 > 0.0 and 1.0 - k1 * z1 / radius >= 0.0
    valid2 = (
        solvable2
        and (resolved_c or resolved_step)
        and t2 > 0.0
        and 1.0 - k1 * z2 / radius >= 0.0
    )
    vertex2 = (abs(z2) if solvable2 else math.inf) < (
        abs(z1) if solvable1 else math.inf
    )
    pick2 = valid2 or (not valid1 and vertex2)
    if solvable1 or solvable2:
        return t2 if pick2 else t1
    return math.nan


class NumpyLens(_Base):
    """Rays vectorized with NumPy; paraxials stay scalar."""

    def __init__(self, snap: Snapshot):
        super().__init__(snap)
        self.px = np.array([p[0] for p in self.pupil])
        self.py = np.array([p[1] for p in self.pupil])

    def trace_pupil(self, epl, hx, hy, w):
        s, z = self.s, self.z
        n = s.index[w]
        x_off, y_off, z0, z_pupil = self.launch_geometry(epl, hx, hy)
        x1 = self.px * s.epd / 2.0
        y1 = self.py * s.epd / 2.0
        x, y = x1 + x_off, y1 + y_off
        zz = np.full_like(x, z0)
        dx, dy, dz = x1 - x, y1 - y, z_pupil - zz
        mag = np.sqrt(dx * dx + dy * dy + dz * dz)
        L, M, N = dx / mag, dy / mag, dz / mag
        with np.errstate(all="ignore"):
            for k in range(1, len(s.kinds)):
                zz = zz - z[k]
                R, kc = s.radius[k], s.conic[k]
                if s.kinds[k] == PLANE:
                    t = -zz / N
                else:
                    t = _conic_distance_np(x, y, zz, L, M, N, R, kc)
                x, y, zz = x + t * L, y + t * M, zz + t * N
                if s.kinds[k] == PLANE:
                    nx, ny, nz = 0.0, 0.0, 1.0
                else:
                    denom = R * np.sqrt(1.0 - (1.0 + kc) * (x * x + y * y) / (R * R))
                    fx, fy = x / denom, y / denom
                    mag = np.sqrt(fx * fx + fy * fy + 1.0)
                    nx, ny, nz = fx / mag, fy / mag, -1.0 / mag
                dot = L * nx + M * ny + N * nz
                sgn = np.sign(dot)
                nx, ny, nz, dot = nx * sgn, ny * sgn, nz * sgn, np.abs(dot)
                u = n[k - 1] / n[k]
                root = np.sqrt(1.0 - u * u * (1.0 - dot * dot))
                L, M, N = (
                    u * L + nx * root - u * nx * dot,
                    u * M + ny * root - u * ny * dot,
                    u * N + nz * root - u * nz * dot,
                )
                zz = zz + z[k]
        return x, y

    def rms_spot(self, epl, hx, hy):
        spots = [self.trace_pupil(epl, hx, hy, w) for w in range(len(self.s.index))]
        xs = np.concatenate([sp[0] for sp in spots])
        ys = np.concatenate([sp[1] for sp in spots])
        if np.isnan(xs).any() or np.isnan(ys).any():
            return math.nan
        cx, cy = spots[self.s.primary][0].mean(), spots[self.s.primary][1].mean()
        return float(np.sqrt(np.mean((xs - cx) ** 2 + (ys - cy) ** 2)))


def _conic_distance_np(x, y, z, L, M, N, radius, conic):
    """Vectorized ``_conic_distance``."""
    k1 = 1.0 + conic
    kz = k1 * z
    transverse = x * x + y * y
    transverse_dir = L * L + M * M
    N2 = N * N
    a = transverse_dir + k1 * N2
    b = 2.0 * (L * x + M * y + N * (kz - radius))
    c = transverse + z * (kz - 2.0 * radius)
    roundoff = 4.0 * EPS
    resolved_c = np.abs(c) > roundoff * (
        transverse + np.abs(z) * (np.abs(kz) + 2 * abs(radius))
    )
    d = b * b - 4.0 * a * c
    sqrt_d = np.where(d > 0, np.sqrt(np.where(d > 0, d, 1.0)), 0.0)
    q = -0.5 * (b + np.copysign(sqrt_d, b))
    a_ok = (a != 0) & (np.abs(a) < np.inf)
    q_ok = (q != 0) & (np.abs(q) < np.inf)
    t1 = q / np.where(a_ok, a, 1.0)
    t2 = c / np.where(q_ok, q, 1.0)
    resolved_step = t2 * t2 * (transverse_dir + N2) > roundoff**2 * (transverse + z * z)
    solvable1 = (d >= 0) & a_ok & (np.abs(t1) < np.inf)
    solvable2 = (d >= 0) & q_ok & (np.abs(t2) < np.inf)
    z1, z2 = z + t1 * N, z + t2 * N
    valid1 = solvable1 & (t1 > 0) & (1 - k1 * z1 / radius >= 0)
    valid2 = (
        solvable2
        & (resolved_c | resolved_step)
        & (t2 > 0)
        & (1 - k1 * z2 / radius >= 0)
    )
    vertex2 = np.where(solvable2, np.abs(z2), np.inf) < np.where(
        solvable1, np.abs(z1), np.inf
    )
    pick2 = valid2 | (~valid1 & vertex2)
    return np.where(solvable1 | solvable2, np.where(pick2, t2, t1), np.nan)


__all__ = ["CONIC", "NumpyLens", "PythonLens"]
