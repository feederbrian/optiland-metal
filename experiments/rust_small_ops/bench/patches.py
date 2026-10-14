"""Reversible prototypes that change how Optiland executes, for measurement.

Each is a context manager that monkeypatches Optiland on entry and restores
it on exit; nothing under ``optiland/`` is edited.

* ``rust_surface_leaf``: tier 2 of the experiment. Replaces the arithmetic
  of ``Surface._trace_real`` (distance, propagate, OPD, refract) for plain
  refractive planes/spheres with one Rust call, leaving all of Optiland's
  orchestration (localize, material lookups, recording, ray aiming) intact.
* ``evaluation_scope``: build-once-per-operation memoization. Each merit
  evaluation and each top-level trace runs in a fresh scope during which
  the lens is treated as frozen, so the paraxial path, first-order pupil
  quantities and refractive indices are computed once and reused.
* ``direct_backend``: binds backend functions into the ``optiland.backend``
  module namespace so ``be.sqrt`` resolves without ``__getattr__`` and the
  passthrough wrapper.
"""

from __future__ import annotations

import contextlib
import contextvars
import functools
import math

import numpy as np
import optiland_small

import optiland.backend as be
from optiland.geometries import Plane, StandardGeometry
from optiland.interactions import RefractiveReflectiveModel
from optiland.materials.base import BaseMaterial
from optiland.optic.optic import Optic
from optiland.optimization.problem import OptimizationProblem
from optiland.paraxial import Paraxial
from optiland.propagation.homogeneous import HomogeneousPropagation
from optiland.rays import RealRays
from optiland.surfaces.standard_surface import Surface
from optiland.surfaces.surface_group import SurfaceGroup


@contextlib.contextmanager
def _patched(owner, name, replacement):
    original = owner.__dict__[name]
    setattr(owner, name, replacement)
    try:
        yield original
    finally:
        setattr(owner, name, original)


# --------------------------------------------------------------------------
# Tier 2: Rust leaf
# --------------------------------------------------------------------------


def _first(value) -> float:
    return float(np.asarray(value).reshape(-1)[0])


def _uniform(values) -> float | None:
    arr = np.asarray(values).reshape(-1)
    first = float(arr[0])
    return first if arr.size == 1 or bool(np.all(arr == first)) else None


@contextlib.contextmanager
def rust_surface_leaf():
    """Route plain refractive plane/sphere surfaces through ``trace_surface``."""
    original = Surface.__dict__["_trace_real"]

    def trace_real(self, rays):
        geometry = self.geometry
        kind = {Plane: 0, StandardGeometry: 1}.get(type(geometry))
        model = self.interaction_model
        wavelength = _uniform(rays.w)
        if (
            kind is None
            or type(rays) is not RealRays
            or not rays.is_normalized
            or type(self.material_pre.propagation_model) is not HomogeneousPropagation
            or self.aperture is not None
            or type(model) is not RefractiveReflectiveModel
            or model.is_reflective
            or model.coating
            or model.bsdf
            or wavelength is None
        ):
            return original(self, rays)
        w = np.array([wavelength])
        n1 = _first(self.material_pre.n(w))
        n2 = _first(self.material_post.n(w))
        k = _first(self.material_pre.k(w))
        alpha = 4 * math.pi * k / wavelength * 1e3 if k > 0 else 0.0
        radius = float(geometry.radius) if kind else math.inf
        conic = float(geometry.k) if kind else 0.0
        size = np.size(rays.x)
        cols = [
            np.ascontiguousarray(np.broadcast_to(getattr(rays, f), (size,)), np.float64)
            for f in ("x", "y", "z", "L", "M", "N", "i", "opd")
        ]
        rays.L0, rays.M0, rays.N0 = cols[3], cols[4], cols[5]
        out = optiland_small.trace_surface(cols, kind, radius, conic, n1, n2, alpha)
        rays.x, rays.y, rays.z, rays.L, rays.M, rays.N, rays.i, rays.opd = out
        rays.update()
        return rays

    with _patched(Surface, "_trace_real", trace_real):
        yield


# --------------------------------------------------------------------------
# Option: build once per operation
# --------------------------------------------------------------------------


_SCOPE: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "frozen_lens_scope", default=None
)


@contextlib.contextmanager
def frozen_lens():
    """One operation during which the lens is held fixed.

    Nested scopes share the outermost cache, so every trace inside one merit
    evaluation reuses the same paraxial path, pupil quantities and indices.
    """
    if _SCOPE.get() is not None:
        yield
        return
    token = _SCOPE.set({})
    try:
        yield
    finally:
        _SCOPE.reset(token)


def _scoped(original, key_fn):
    """Memoize ``original`` in the active scope; pass through outside one."""

    @functools.wraps(original)
    def wrapper(*args, **kwargs):
        cache = _SCOPE.get()
        key = None if cache is None else key_fn(*args, **kwargs)
        if key is None:
            return original(*args, **kwargs)
        if key not in cache:
            cache[key] = original(*args, **kwargs)
        return cache[key]

    return wrapper


def _run_in_scope(original):
    @functools.wraps(original)
    def wrapper(*args, **kwargs):
        with frozen_lens():
            return original(*args, **kwargs)

    return wrapper


@contextlib.contextmanager
def evaluation_scope():
    """Build-once memoization, with a fresh cache per operation.

    A scope opens around every ``OptimizationProblem.sum_squared`` (after the
    optimizer has applied its variables) and every top-level
    ``Optic.trace`` / ``Optic.trace_generic``; nothing is reused across
    operations, so a lens changed between calls is always seen.
    """

    def first_order_key(name):
        return lambda self, *a, **kw: None if a or kw else ("fo", id(self.optic), name)

    def index_wrapper(name):
        original = BaseMaterial.__dict__[name]

        def compute(self, value):
            return _first(original(self, np.array([value])))

        cached = _scoped(compute, lambda self, value: ("n", id(self), name, value))

        def wrapper(self, wavelength, **kwargs):
            value = None if kwargs else _uniform_or_none(wavelength)
            if value is None or _SCOPE.get() is None:
                return original(self, wavelength, **kwargs)
            result = cached(self, value)
            if np.ndim(wavelength) == 0:
                return np.asarray(result)
            return np.full(np.shape(wavelength), result)

        return wrapper

    with contextlib.ExitStack() as stack:
        build = SurfaceGroup.__dict__["build_paraxial_path"]
        stack.enter_context(
            _patched(
                SurfaceGroup,
                "build_paraxial_path",
                _scoped(build, lambda self: ("path", id(self))),
            )
        )
        for name in ("EPD", "EPL", "entrance_pupil_axial_position", "f2", "XPL"):
            original = Paraxial.__dict__[name]
            stack.enter_context(
                _patched(Paraxial, name, _scoped(original, first_order_key(name)))
            )
        for name in ("n", "k"):
            stack.enter_context(_patched(BaseMaterial, name, index_wrapper(name)))
        for owner, name in (
            (OptimizationProblem, "sum_squared"),
            (Optic, "trace"),
            (Optic, "trace_generic"),
        ):
            stack.enter_context(
                _patched(owner, name, _run_in_scope(owner.__dict__[name]))
            )
        yield


def _uniform_or_none(wavelength) -> float | None:
    if isinstance(wavelength, float | int):
        return float(wavelength)
    if isinstance(wavelength, np.ndarray) and wavelength.dtype == np.float64:
        flat = wavelength.reshape(-1)
        if flat.size and (flat.size == 1 or flat.min() == flat.max()):
            return float(flat[0])
    return None


# --------------------------------------------------------------------------
# Option: direct backend binding
# --------------------------------------------------------------------------


@contextlib.contextmanager
def direct_backend():
    """Resolve ``be.<name>`` as a plain module attribute.

    Passthrough methods are replaced by the NumPy function they forward to;
    every other backend method is bound once. Restored on exit.
    """
    instance = be._backends[be.get_backend()]
    module_globals = vars(be)
    added = []
    for name in dir(type(instance)):
        if name.startswith("_") or name in module_globals:
            continue
        attr = getattr(type(instance), name)
        if not callable(attr):
            continue
        qual = getattr(attr, "__qualname__", "")
        lib_fn = (
            getattr(instance._lib, name, None) if hasattr(instance, "_lib") else None
        )
        code = getattr(attr, "__code__", None)
        is_passthrough = qual.endswith(f".{name}") and code and code.co_name == "method"
        module_globals[name] = (
            lib_fn if is_passthrough and lib_fn else getattr(instance, name)
        )
        added.append(name)
    try:
        yield
    finally:
        for name in added:
            module_globals.pop(name, None)
