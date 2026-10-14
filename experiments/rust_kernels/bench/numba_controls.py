"""Numba versions of the Rust kernels, written to the same algorithm.

These are the control group: when Rust beats Optiland, comparing against
these tells us whether the gain comes from Rust itself or simply from fusing
the computation into one compiled loop, which Numba (already an Optiland
dependency) could do too.

Both tracers mirror ``src/trace.rs`` statement for statement; the Huygens
kernel mirrors the ``hoisted`` variant in ``src/huygens.rs``.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

EPS = np.finfo(np.float64).eps
DENOM_EPS = 32.0 * EPS
CONV_EPS = 8.0 * EPS
PLANE, CONIC, EVEN_ASPHERE = 0, 1, 2


@njit(cache=True, inline="always", error_model="numpy")
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
    d_ok = d >= 0.0
    sqrt_d = math.sqrt(d) if d > 0.0 else 0.0
    q = -0.5 * (b + math.copysign(sqrt_d, b))
    a_ok = a != 0.0 and abs(a) < math.inf
    q_ok = q != 0.0 and abs(q) < math.inf
    t1 = q / (a if a_ok else 1.0)
    t2 = c / (q if q_ok else 1.0)
    resolved_step = t2 * t2 * (transverse_dir + N2) > roundoff * roundoff * (
        transverse + z * z
    )
    solvable1 = d_ok and a_ok and abs(t1) < math.inf
    solvable2 = d_ok and q_ok and abs(t2) < math.inf
    z1 = z + t1 * N
    z2 = z + t2 * N
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


@njit(cache=True, inline="always", error_model="numpy")
def _asphere_sag(x, y, radius, conic, coeffs, ncoef):
    r2 = x * x + y * y
    base = r2 / (
        radius * (1.0 + math.sqrt(1.0 - (1.0 + conic) * r2 / (radius * radius)))
    )
    poly = 0.0
    for i in range(ncoef - 1, -1, -1):
        poly = poly * r2 + coeffs[i]
    return base + poly * r2


@njit(cache=True, inline="always", error_model="numpy")
def _slopes(x, y, radius, conic, coeffs, ncoef):
    r2 = x * x + y * y
    denom = radius * math.sqrt(1.0 - (1.0 + conic) * r2 / (radius * radius))
    dpoly = 0.0
    for i in range(ncoef - 1, -1, -1):
        dpoly = dpoly * r2 + (i + 1.0) * coeffs[i]
    return x / denom + 2.0 * x * dpoly, y / denom + 2.0 * y * dpoly


@njit(cache=True, inline="always", error_model="numpy")
def _asphere_distance(x, y, z, L, M, N, radius, conic, coeffs, ncoef, tol, max_iter):
    t = _conic_distance(x, y, z, L, M, N, radius, conic)
    tol = max(tol, CONV_EPS * max(abs(t), 1.0))
    f = _asphere_sag(x + t * L, y + t * M, radius, conic, coeffs, ncoef) - (z + t * N)
    it = 0
    while not (abs(f) < tol) and it < max_iter:
        sx, sy = _slopes(x + t * L, y + t * M, radius, conic, coeffs, ncoef)
        mag = math.sqrt(sx * sx + sy * sy + 1.0)
        nx, ny, nz = sx / mag, sy / mag, -1.0 / mag
        if abs(nz) <= DENOM_EPS:
            nz = DENOM_EPS if nz >= 0.0 else -DENOM_EPS
        fx, fy = -nx / nz, -ny / nz
        df_dt = fx * L + fy * M - N
        tau = DENOM_EPS * max(abs(fx * L) + abs(fy * M) + abs(N), 1.0)
        if abs(df_dt) <= tau:
            df_dt = tau if df_dt >= 0.0 else -tau
        t -= f / df_dt
        it += 1
        f = _asphere_sag(x + t * L, y + t * M, radius, conic, coeffs, ncoef) - (
            z + t * N
        )
    return t


@njit(cache=True, parallel=True, error_model="numpy")
def trace_sequential(
    x, y, z, L, M, N, i, opd,
    kinds, vertex_z, radius, conic, coefficients, n_pre, n_post, alpha_pre,
    tol, max_iter,
):  # fmt: skip
    """Fused ray-major trace; returns new arrays like the Rust kernel."""
    x, y, z, L, M, N, i, opd = (
        x.copy(),
        y.copy(),
        z.copy(),
        L.copy(),
        M.copy(),
        N.copy(),
        i.copy(),
        opd.copy(),
    )
    nsurf = kinds.shape[0]
    ncoefs = np.zeros(nsurf, dtype=np.int64)
    for s in range(nsurf):
        n = coefficients.shape[1]
        while n > 0 and coefficients[s, n - 1] == 0.0:
            n -= 1
        ncoefs[s] = n

    for k in prange(x.shape[0]):
        rx, ry, rz, rl, rm, rn, ri, ro = (
            x[k],
            y[k],
            z[k],
            L[k],
            M[k],
            N[k],
            i[k],
            opd[k],
        )
        for s in range(nsurf):
            kind, R, kc = kinds[s], radius[s], conic[s]
            rz -= vertex_z[s]
            if kind == PLANE:
                t = -rz / rn
            elif kind == CONIC:
                t = _conic_distance(rx, ry, rz, rl, rm, rn, R, kc)
            else:
                t = _asphere_distance(
                    rx,
                    ry,
                    rz,
                    rl,
                    rm,
                    rn,
                    R,
                    kc,
                    coefficients[s],
                    ncoefs[s],
                    tol,
                    max_iter,
                )
            rx += t * rl
            ry += t * rm
            rz += t * rn
            if alpha_pre[s] > 0.0:
                ri *= math.exp(-alpha_pre[s] * t)
            ro += t * n_pre[s]

            if kind == PLANE:
                nx, ny, nz = 0.0, 0.0, 1.0
            else:
                if kind == CONIC:
                    r2 = rx * rx + ry * ry
                    denom = R * math.sqrt(1.0 - (1.0 + kc) * r2 / (R * R))
                    fx, fy = rx / denom, ry / denom
                else:
                    fx, fy = _slopes(rx, ry, R, kc, coefficients[s], ncoefs[s])
                mag = math.sqrt(fx * fx + fy * fy + 1.0)
                nx, ny, nz = fx / mag, fy / mag, -1.0 / mag

            dot = rl * nx + rm * ny + rn * nz
            sgn = np.sign(dot)
            nx, ny, nz, dot = nx * sgn, ny * sgn, nz * sgn, abs(dot)
            u = n_pre[s] / n_post[s]
            root = math.sqrt(1.0 - u * u * (1.0 - dot * dot))
            rl, rm, rn = (
                u * rl + nx * root - u * nx * dot,
                u * rm + ny * root - u * ny * dot,
                u * rn + nz * root - u * nz * dot,
            )
            rz += vertex_z[s]
        x[k], y[k], z[k], L[k], M[k], N[k], i[k], opd[k] = (
            rx,
            ry,
            rz,
            rl,
            rm,
            rn,
            ri,
            ro,
        )
    return x, y, z, L, M, N, i, opd


@njit(cache=True, parallel=True, fastmath=True, error_model="numpy")
def huygens_hoisted(
    image_x, image_y, image_z, px, py, pz, amp, opd, wavelength, rp, center
):  # fmt: skip
    """Hoisted Huygens-Fresnel sum: one complex exponential per pair."""
    k = 2.0 * np.pi / wavelength
    n = px.shape[0]
    wr = np.empty(n)
    wi = np.empty(n)
    for j in range(n):
        wr[j] = amp[j] * math.cos(-k * opd[j])
        wi[j] = amp[j] * math.sin(-k * opd[j])
    # Inward reference-sphere normal, as in Optiland since #864.
    nx = (center[0] - px) / rp
    ny = (center[1] - py) / rp
    nz = (center[2] - pz) / rp

    flat_x, flat_y, flat_z = image_x.ravel(), image_y.ravel(), image_z.ravel()
    out = np.empty(flat_x.shape[0])
    for p in prange(flat_x.shape[0]):
        x, y, z = flat_x[p], flat_y[p], flat_z[p]
        re = 0.0
        im = 0.0
        for j in range(n):
            dx, dy, dz = x - px[j], y - py[j], z - pz[j]
            r = math.sqrt(dx * dx + dy * dy + dz * dz)
            inv_r = 1.0 / r
            scale = 0.5 * (1.0 + (dx * nx[j] + dy * ny[j] + dz * nz[j]) * inv_r) * inv_r
            c, s = math.cos(k * r), math.sin(k * r)
            re += scale * (wr[j] * c - wi[j] * s)
            im += scale * (wr[j] * s + wi[j] * c)
        out[p] = re * re + im * im
    return out.reshape(image_x.shape)


# fdlibm's 33-bit split of pi/2: n * PIO2_1 is exact for |n| < 2**20, which
# Numba needs because it cannot express FMA (no ``math.fma``).
FRAC_2_PI = 2.0 / np.pi
PIO2_1 = 1.57079632673412561417e00
PIO2_1T = 6.07710050650619224932e-11
FAST_SINCOS_LIMIT = 2.0**20 * PIO2_1
_S = (
    -1.66666666666666324348e-01, 8.33333333332248946124e-03,
    -1.98412698298579493134e-04, 2.75573137070700676789e-06,
    -2.50507602534068634195e-08, 1.58969099521155010221e-10,
)  # fmt: skip
_C = (
    4.16666666666666019037e-02, -1.38888888888741095749e-03,
    2.48015872894767294178e-05, -2.75573143513906633035e-07,
    2.08757232129817482790e-09, -1.13596475577881948265e-11,
)  # fmt: skip

# Every fastmath flag except ``reassoc``, which would fold the Cody-Waite
# reduction ``(x - n*P1) - n*P1T`` back into an inaccurate single step.
_NO_REASSOC = {"nnan", "ninf", "nsz", "arcp", "contract", "afn"}


@njit(cache=True, inline="always", error_model="numpy", fastmath=_NO_REASSOC)
def _fast_sincos(x):
    """Same reduction and fdlibm kernels as ``src/fastmath.rs``, branch-free."""
    n = math.floor(x * FRAC_2_PI + 0.5)
    q = np.int64(n)
    r = (x - n * PIO2_1) - n * PIO2_1T
    z = r * r
    s = r + r * z * (
        _S[0] + z * (_S[1] + z * (_S[2] + z * (_S[3] + z * (_S[4] + z * _S[5]))))
    )
    c = (
        1.0
        - 0.5 * z
        + z
        * z
        * (_C[0] + z * (_C[1] + z * (_C[2] + z * (_C[3] + z * (_C[4] + z * _C[5])))))
    )
    swap = (q & 1) != 0
    sin_sign = 1.0 - (q & 2)
    cos_sign = 1.0 - ((q + 1) & 2)
    return sin_sign * (c if swap else s), cos_sign * (s if swap else c)


LANES = 8


@njit(cache=True, parallel=True, error_model="numpy", fastmath=_NO_REASSOC)
def huygens_simd(
    image_x, image_y, image_z, px, py, pz, amp, opd, wavelength, rp, center
):  # fmt: skip
    """Numba mirror of the Rust ``simd`` variant (8-lane accumulators)."""
    k = 2.0 * np.pi / wavelength
    n = px.shape[0]
    padded = (n + LANES - 1) // LANES * LANES
    x_, y_, z_ = (
        np.full(padded, px[-1]),
        np.full(padded, py[-1]),
        np.full(padded, pz[-1]),
    )
    x_[:n], y_[:n], z_[:n] = px, py, pz
    wr, wi = np.zeros(padded), np.zeros(padded)
    for j in range(n):
        wr[j] = amp[j] * math.cos(-k * opd[j])
        wi[j] = amp[j] * math.sin(-k * opd[j])
    nx = (center[0] - x_) / rp
    ny = (center[1] - y_) / rp
    nz = (center[2] - z_) / rp

    flat_x, flat_y, flat_z = image_x.ravel(), image_y.ravel(), image_z.ravel()
    out = np.empty(flat_x.shape[0])
    for p in prange(flat_x.shape[0]):
        x, y, z = flat_x[p], flat_y[p], flat_z[p]
        re = np.zeros(LANES)
        im = np.zeros(LANES)
        for base in range(0, padded, LANES):
            for lane in range(LANES):
                j = base + lane
                dx, dy, dz = x - x_[j], y - y_[j], z - z_[j]
                r = math.sqrt(dx * dx + dy * dy + dz * dz)
                inv_r = 1.0 / r
                scale = (
                    0.5 * (1.0 + (dx * nx[j] + dy * ny[j] + dz * nz[j]) * inv_r) * inv_r
                )
                s, c = _fast_sincos(k * r)
                re[lane] += scale * (wr[j] * c - wi[j] * s)
                im[lane] += scale * (wr[j] * s + wi[j] * c)
        tr, ti = re.sum(), im.sum()
        out[p] = tr * tr + ti * ti
    return out.reshape(image_x.shape)


def max_phase(image, pupil, wavelength) -> float:
    """Upper bound on k*R over all image/pupil pairs (bounding-box corners)."""
    span = [
        max(abs(i.max() - p.min()), abs(p.max() - i.min()))
        for i, p in zip(image, pupil, strict=True)
    ]
    return 2.0 * np.pi / wavelength * float(np.sqrt(sum(s * s for s in span)))
