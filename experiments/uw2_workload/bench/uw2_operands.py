"""Proxy for one evaluation of the UW2 V2 merit function, from public Optiland calls.

The pipeline's own operand code is not available, so each operand kind in
``V2.json`` ("problem") is reproduced with the Optiland call it most likely
rests on, at the sampling the job records:

* ``wave``  (20): RMS wavefront error at u = 0, 0.5, 0.7, 1 over 5 wavelengths,
  6-ring hexapolar pupil, chief-ray reference sphere.
* ``spot``  (4): RMS spot radius at the same fields over 5 wavelengths, rays
  blocked by clear apertures excluded (the pipeline's "designed-blocked").
* ``chief`` (frame, f_eq, CRA): chief rays at a short ladder of field
  angles, as a root-find for the framing angle does.
* ``surv``  (1): the corner bundle, 8 rings over 5 wavelengths.
* ``dual``  (7): the F/2.8 configuration: spots at u = 0, 0.7, 1 with
  12 rings and wavefronts at u = 0, 0.7 (d-line), then restore F/1.8.
* ``sens``  (32): 16 tolerance perturbations x 2 fields of d-line wavefront.
  Thickness perturbations stand in for the decenters the job lists.

The point is to reproduce where Optiland spends the time for this lens, not
the pipeline's residual values.
"""

from __future__ import annotations

import numpy as np

import optiland.backend as be
from optiland.wavefront import Wavefront

WAVE_WLS = (0.4358, 0.4861, 0.5461, 0.5876, 0.6563)
D_LINE = 0.5876
MERIT_US = (0.0, 0.5, 0.7, 1.0)
RINGS = 6
SURV_RINGS = 8
DUAL_FNO = 2.8
DUAL_RINGS = 12
FRAME_US = (0.8, 0.9, 0.95, 0.975, 1.0)
# Thicknesses perturbed by the sensitivity proxy (gap index -> step mm).
SENS_GAPS = (2, 4, 7, 9, 11, 13, 15, 17, 19, 20, 21, 22, 23, 24, 25, 26)
SENS_STEP_MM = 0.01


def _np(a) -> np.ndarray:
    return np.asarray(be.to_numpy(a), dtype=np.float64).ravel()


def wave_rms(lens, u: float, wl: float, rings: int = RINGS) -> float:
    """RMS wavefront error in waves over unblocked rays (piston removed)."""
    wf = Wavefront(
        lens,
        fields=[(0.0, u)],
        wavelengths=[wl],
        num_rays=rings,
        distribution="hexapolar",
        strategy="chief_ray",
    )
    data = wf.get_data((0.0, u), wl)
    opd, inten = _np(data.opd), _np(data.intensity)
    ok = np.isfinite(opd) & (inten > 0)
    if not ok.any():
        return float("nan")
    v = opd[ok]
    return float(np.sqrt(np.mean((v - v.mean()) ** 2)))


def spot_rms(lens, u: float, wls=WAVE_WLS, rings: int = RINGS) -> float:
    """Polychromatic RMS spot (mm) about the primary-wavelength centroid."""
    xs, ys, prim = [], [], None
    for wl in wls:
        rays = lens.trace(0.0, u, wl, rings, "hexapolar", record=False)
        x, y, i = _np(rays.x), _np(rays.y), _np(rays.i)
        ok = np.isfinite(x) & np.isfinite(y) & (i > 0)
        xs.append(x[ok])
        ys.append(y[ok])
        if abs(wl - D_LINE) < 1e-9:
            prim = (x[ok].mean(), y[ok].mean())
    cx, cy = prim if prim is not None else (np.mean(xs[0]), np.mean(ys[0]))
    x, y = np.concatenate(xs), np.concatenate(ys)
    return float(np.sqrt(np.mean((x - cx) ** 2 + (y - cy) ** 2)))


def chief(lens, u: float, wl: float = D_LINE) -> tuple[float, float]:
    """Chief-ray image height (mm) and angle (deg)."""
    r = lens.trace_generic(0.0, u, 0.0, 0.0, wl)
    n = float(_np(r.N)[0])
    return float(_np(r.y)[0]), float(np.degrees(np.arccos(min(1.0, abs(n)))))


def survival(lens, u: float = 1.0, wls=WAVE_WLS, rings: int = SURV_RINGS) -> float:
    """Fraction of corner rays that arrive (finite) or are cleanly blocked."""
    alive = total = 0
    for wl in wls:
        rays = lens.trace(0.0, u, wl, rings, "hexapolar", record=False)
        x, i = _np(rays.x), _np(rays.i)
        total += x.size
        alive += int(np.sum(np.isfinite(x) | (i == 0)))
    return alive / total


def _set_fno(lens, fno: float) -> None:
    lens.set_aperture(aperture_type="imageFNO", value=fno)


def _thickness(lens, gap: int) -> float:
    return float(_np(lens.surfaces.surfaces[gap].thickness)[0])


def _set_thickness(lens, gap: int, value: float) -> None:
    lens.updater.set_thickness(value, gap)


def evaluate(lens, parts=None) -> dict[str, float]:
    """One proxy merit evaluation. Returns each operand kind's values."""
    parts = parts or ("wave", "spot", "chief", "surv", "dual", "sens")
    out: dict[str, list] = {p: [] for p in parts}
    if "wave" in parts:
        out["wave"] = [wave_rms(lens, u, wl) for u in MERIT_US for wl in WAVE_WLS]
    if "spot" in parts:
        out["spot"] = [spot_rms(lens, u) for u in MERIT_US]
    if "chief" in parts:
        out["chief"] = [v for u in FRAME_US for v in chief(lens, u)]
    if "surv" in parts:
        out["surv"] = [survival(lens)]
    if "dual" in parts:
        _set_fno(lens, DUAL_FNO)
        try:
            out["dual"] = [spot_rms(lens, u, rings=DUAL_RINGS) for u in (0.0, 0.7, 1.0)]
            out["dual"] += [wave_rms(lens, u, D_LINE) for u in (0.0, 0.7)]
        finally:
            _set_fno(lens, 1.8)
    if "sens" in parts:
        nominal = {u: wave_rms(lens, u, D_LINE) for u in (0.0, 0.7)}
        for gap in SENS_GAPS:
            t0 = _thickness(lens, gap)
            _set_thickness(lens, gap, t0 + SENS_STEP_MM)
            try:
                out["sens"] += [
                    wave_rms(lens, u, D_LINE) - nominal[u] for u in (0.0, 0.7)
                ]
            finally:
                _set_thickness(lens, gap, t0)
    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}
