"""Shared fixtures for the Rust kernel experiment.

Builds the benchmark lenses, extracts flat surface tables for the compiled
tracers, captures real Huygens-Fresnel inputs from Optiland, and provides the
Optiland reference implementations that every compiled variant is checked
against.
"""

from __future__ import annotations

import copy
import time
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

import optiland.backend as be
from optiland import optic as optic_module
from optiland.geometries import EvenAsphere, Plane, StandardGeometry
from optiland.interactions import RefractiveReflectiveModel
from optiland.psf import huygens_fresnel_strategies as hfs
from optiland.psf.huygens_fresnel import ScalarHuygensPSF
from optiland.samples.objectives import DoubleGauss
from optiland.samples.simple import AsphericSinglet

if TYPE_CHECKING:
    from collections.abc import Callable

warnings.filterwarnings("ignore")
be.set_backend("numpy")

PLANE, CONIC, EVEN_ASPHERE = 0, 1, 2
RAY_FIELDS = ("x", "y", "z", "L", "M", "N", "i", "opd")


# --------------------------------------------------------------------------
# Lenses
# --------------------------------------------------------------------------


def aspheric_double_gauss() -> optic_module.Optic:
    """The Double Gauss sample with every curved surface made an even asphere.

    Small 4th-8th order terms (tens of microns of departure at the edge) keep
    the design tracing cleanly while forcing real Newton-Raphson iterations
    on all eight curved surfaces; the stock aspheric singlet converges in
    about one step.
    """
    lens = optic_module.Optic()
    rows = [
        (56.20238, 8.75, "N-SSK2"),
        (152.28580, 0.5, "air"),
        (37.68262, 12.5, "N-SK2"),
        (be.inf, 3.8, ("F5", "schott")),
        (24.23130, 16.369445, "air"),
        (be.inf, 13.747957, "air"),
        (-28.37731, 3.8, ("F5", "schott")),
        (be.inf, 11, "N-SK16"),
        (-37.92546, 0.5, "air"),
        (177.41176, 7, "N-SK16"),
        (-79.41143, 61.487536, "air"),
    ]
    lens.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    for index, (radius, thickness, material) in enumerate(rows, start=1):
        kwargs = {"index": index, "radius": radius, "thickness": thickness}
        if material != "air":
            kwargs["material"] = material
        if index == 6:
            kwargs["is_stop"] = True
        if np.isfinite(radius):
            sign = 1.0 if index % 2 else -1.0
            kwargs["surface_type"] = "even_asphere"
            kwargs["coefficients"] = [sign * 2e-6, -sign * 1e-9, sign * 5e-13]
        lens.surfaces.add(**kwargs)
    lens.surfaces.add(index=len(rows) + 1)
    lens.set_aperture(aperture_type="imageFNO", value=5)
    lens.fields.set_type(field_type="angle")
    lens.fields.add(y=0)
    lens.fields.add(y=10)
    lens.fields.add(y=14)
    lens.wavelengths.add(value=0.5876, is_primary=True)
    return lens


LENSES: dict[str, Callable[[], optic_module.Optic]] = {
    "double_gauss": DoubleGauss,
    "aspheric_singlet": AsphericSinglet,
    "aspheric_double_gauss": aspheric_double_gauss,
}


# --------------------------------------------------------------------------
# Sequential trace
# --------------------------------------------------------------------------


@dataclass
class SurfaceTable:
    """Flat per-surface parameters consumed by the Rust and Numba tracers."""

    kinds: np.ndarray
    vertex_z: np.ndarray
    radius: np.ndarray
    conic: np.ndarray
    coefficients: np.ndarray
    n_pre: np.ndarray
    n_post: np.ndarray
    alpha_pre: np.ndarray
    tol: float
    max_iter: int

    def args(self) -> tuple:
        return (
            self.kinds,
            self.vertex_z,
            self.radius,
            self.conic,
            self.coefficients,
            self.n_pre,
            self.n_post,
            self.alpha_pre,
        )


def _scalar(value) -> float:
    return float(np.asarray(be.to_numpy(value)).reshape(-1)[0])


def surface_table(lens: optic_module.Optic, wavelength: float) -> SurfaceTable:
    """Flatten surfaces 1..N of a centered, refractive, unapertured system.

    Raises:
        NotImplementedError: For anything the experimental tracers do not
            model (tilts, decenters, apertures, mirrors, coatings, other
            geometry types).
    """
    rows, tol, max_iter = [], 1e-10, 100
    for surface in lens.surfaces.surfaces[1:]:
        geometry, cs = surface.geometry, surface.geometry.cs
        if any(_scalar(v) != 0.0 for v in (cs.x, cs.y, cs.rx, cs.ry, cs.rz)):
            raise NotImplementedError("tilted or decentered surface")
        if surface.aperture is not None:
            raise NotImplementedError("physical aperture")
        model = surface.interaction_model
        if type(model) is not RefractiveReflectiveModel or model.is_reflective:
            raise NotImplementedError("only refraction is modeled")
        if model.coating or model.bsdf:
            raise NotImplementedError("coatings and BSDFs are not modeled")

        coefficients: list[float] = []
        if type(geometry) is Plane:
            kind, radius, conic = PLANE, np.inf, 0.0
        elif type(geometry) is EvenAsphere:
            kind, radius, conic = EVEN_ASPHERE, geometry.radius, geometry.k
            coefficients = [_scalar(c) for c in geometry.coefficients]
            tol, max_iter = float(geometry.tol), int(geometry.max_iter)
        elif type(geometry) is StandardGeometry:
            kind, radius, conic = CONIC, geometry.radius, geometry.k
        else:
            raise NotImplementedError(type(geometry).__name__)

        w = be.array([wavelength])
        k_ext = _scalar(surface.material_pre.k(w))
        alpha = 4 * np.pi * k_ext / wavelength * 1e3 if k_ext > 0 else 0.0
        rows.append(
            (
                kind,
                _scalar(cs.z),
                _scalar(radius),
                _scalar(conic),
                coefficients,
                _scalar(surface.material_pre.n(w)),
                _scalar(surface.material_post.n(w)),
                alpha,
            )
        )

    width = max(1, max(len(r[4]) for r in rows))
    coeffs = np.zeros((len(rows), width))
    for s, row in enumerate(rows):
        coeffs[s, : len(row[4])] = row[4]
    col = lambda j, dtype=np.float64: np.array([r[j] for r in rows], dtype=dtype)  # noqa: E731
    return SurfaceTable(
        kinds=col(0, np.int64),
        vertex_z=col(1),
        radius=col(2),
        conic=col(3),
        coefficients=coeffs,
        n_pre=col(5),
        n_post=col(6),
        alpha_pre=col(7),
        tol=tol,
        max_iter=max_iter,
    )


def launch_rays(lens, num_rays: int, Hy: float = 1.0, wavelength: float | None = None):
    """Rays as Optiland launches them for ``lens.trace`` (uniform pupil grid)."""
    from optiland.distribution import create_distribution  # noqa: PLC0415

    wavelength = wavelength or lens.primary_wavelength
    dist = create_distribution("uniform")
    dist.generate_points(num_rays)
    n = len(dist.x)
    rays = lens.ray_tracer.ray_generator.generate_rays(
        np.zeros(n), np.full(n, Hy), dist.x, dist.y, wavelength
    )
    return rays


def ray_columns(rays) -> list[np.ndarray]:
    """Eight contiguous float64 columns in the order the kernels expect."""
    return [
        np.ascontiguousarray(
            np.broadcast_to(getattr(rays, f), rays.x.shape), np.float64
        )
        for f in RAY_FIELDS
    ]


def trace_optiland(lens, rays) -> list[np.ndarray]:
    """Optiland's own NumPy surface loop on a copy of ``rays``."""
    work = copy.deepcopy(rays)
    lens.surfaces.trace(work, skip=1, record=False)
    return ray_columns(work)


# --------------------------------------------------------------------------
# Huygens-Fresnel
# --------------------------------------------------------------------------


def capture_huygens_inputs(lens, num_rays: int, image_size: int, Hy: float = 1.0):
    """Arguments Optiland passes to ``NumbaSummation.compute`` for its PSF.

    Wraps the strategy while a ``ScalarHuygensPSF`` is built and keeps the
    largest call (the second call computes the ideal-PSF normalization).
    """
    calls = []
    original = hfs.NumbaSummation.compute

    def spy(self, *args):
        calls.append(
            tuple(np.asarray(be.to_numpy(a)) if np.ndim(a) else float(a) for a in args)
        )
        return original(self, *args)

    hfs.NumbaSummation.compute = spy
    try:
        ScalarHuygensPSF(
            lens,
            (0, Hy),
            lens.primary_wavelength,
            num_rays=num_rays,
            image_size=image_size,
        )
    finally:
        hfs.NumbaSummation.compute = original
    args = max(calls, key=lambda c: np.size(c[0]))
    image = tuple(np.ascontiguousarray(a, np.float64) for a in args[:3])
    pupil = tuple(np.ascontiguousarray(a, np.float64).ravel() for a in args[3:8])
    return image, pupil, float(args[8]), float(args[9])


def huygens_optiland(image, pupil, wavelength, rp) -> np.ndarray:
    """Optiland's Numba kernel, called exactly as the PSF class calls it."""
    return hfs.NumbaSummation._huygens_fresnel_summation(*image, *pupil, wavelength, rp)


# --------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------


def best_of(fn: Callable[[], object], repeats: int = 5, min_time: float = 0.2) -> float:
    """Best wall time of ``repeats`` runs after a warm-up (JIT, caches)."""
    fn()
    times = []
    deadline = time.perf_counter() + 60.0
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
        if sum(times) > min_time and len(times) >= 3 or time.perf_counter() > deadline:
            break
    return min(times)
