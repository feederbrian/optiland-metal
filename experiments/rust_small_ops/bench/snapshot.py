"""Flatten an Optiland optic + optimization problem into plain data.

The snapshot holds what the merit evaluation needs and nothing else. It is
built once; afterwards only the variable vector changes. Every compiled-lens
implementation (Rust, Numba, plain Python, lean NumPy) consumes it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

import optiland.backend as be
from optiland.geometries import Plane, StandardGeometry
from optiland.interactions import RefractiveReflectiveModel
from optiland.optimization.scaling.linear import LinearScaler

PLANE, CONIC = 0, 1


@dataclass
class Snapshot:
    """Plain-data view of a lens and its merit function."""

    kinds: list[int]
    radius: list[float]
    conic: list[float]
    thickness: list[float]
    index: list[list[float]]  # [wavelength][surface], index after the surface
    first_z: float
    stop: int
    epd: float
    max_field: float
    primary: int
    num_rings: int
    f2_weight: float = 0.0
    f2_target: float = 0.0
    spots: list[tuple[float, float, float, float]] = field(default_factory=list)
    variables: list[tuple[str, int, float, float]] = field(default_factory=list)


def _f(value) -> float:
    return float(np.asarray(be.to_numpy(value)).reshape(-1)[0])


def capture(lens, problem=None) -> Snapshot:
    """Snapshot ``lens`` (and ``problem``'s operands/variables, if given).

    Raises:
        NotImplementedError: For anything outside the experiment's scope.
    """
    surfaces = lens.surfaces.surfaces
    if not lens.object_surface.is_infinite:
        raise NotImplementedError("finite object")
    if type(lens.fields.field_definition).__name__ != "AngleField":
        raise NotImplementedError("only angle fields")
    if lens.aperture.ap_type != "EPD":
        raise NotImplementedError("only EPD apertures")
    if any(_f(f.vx) or _f(f.vy) for f in lens.fields.fields):
        raise NotImplementedError("vignetting factors")

    kinds, radius, conic, thickness = [], [], [], []
    for k, surface in enumerate(surfaces):
        geometry, cs = surface.geometry, surface.geometry.cs
        if k and any(_f(v) != 0.0 for v in (cs.x, cs.y, cs.rx, cs.ry, cs.rz)):
            raise NotImplementedError("tilted or decentered surface")
        if surface.aperture is not None:
            raise NotImplementedError("physical aperture")
        model = surface.interaction_model
        if k and (type(model) is not RefractiveReflectiveModel or model.is_reflective):
            raise NotImplementedError("only refraction")
        if k and (model.coating or model.bsdf):
            raise NotImplementedError("coatings / BSDF")
        if type(geometry) is Plane:
            kinds.append(PLANE)
            radius.append(math.inf)
            conic.append(0.0)
        elif type(geometry) is StandardGeometry:
            kinds.append(CONIC)
            radius.append(_f(geometry.radius))
            conic.append(_f(geometry.k))
        else:
            raise NotImplementedError(type(geometry).__name__)
        thickness.append(_f(surface.thickness) if k else math.inf)

    index = [
        [_f(s.material_post.n(be.array([w]))) for s in surfaces]
        for w in lens.wavelengths.get_wavelengths()
    ]
    snap = Snapshot(
        kinds=kinds,
        radius=radius,
        conic=conic,
        thickness=thickness,
        index=index,
        first_z=_f(surfaces[1].geometry.cs.z),
        stop=int(lens.surfaces.stop_index),
        epd=_f(lens.paraxial.EPD()),
        max_field=_f(lens.fields.max_field),
        primary=int(lens.wavelengths.primary_index),
        num_rings=0,
    )
    if problem is not None:
        _capture_problem(snap, lens, problem)
    return snap


def _capture_problem(snap: Snapshot, lens, problem) -> None:
    rings = set()
    for op in problem.operands:
        data = op.input_data
        if data.get("optic") is not lens:
            raise NotImplementedError("operand on another optic")
        if op.min_val is not None or op.max_val is not None:
            raise NotImplementedError("bounded operands")
        if op.operand_type == "f2":
            snap.f2_weight, snap.f2_target = float(op.weight), float(op.target)
        elif op.operand_type == "rms_spot_size":
            if data.get("wavelength") != "all" or data.get("surface_number") != -1:
                raise NotImplementedError("rms_spot_size other than all-wl image")
            if data.get("distribution", "hexapolar") != "hexapolar":
                raise NotImplementedError("non-hexapolar distribution")
            rings.add(int(data["num_rays"]))
            snap.spots.append(
                (
                    float(data["Hx"]),
                    float(data["Hy"]),
                    float(op.weight),
                    float(op.target),
                )
            )
        else:
            raise NotImplementedError(op.operand_type)
    if len(rings) > 1:
        raise NotImplementedError("mixed ring counts")
    snap.num_rings = rings.pop() if rings else 0

    for var in problem.variables:
        scaler = var.variable.scaler
        if type(scaler) is not LinearScaler or var.type not in ("radius", "thickness"):
            raise NotImplementedError(f"{var.type} with {type(scaler).__name__}")
        surface = int(var.kwargs["surface_number"])
        snap.variables.append(
            (var.type, surface, float(scaler.factor), float(scaler.offset))
        )


def to_rust(snap: Snapshot):
    """Build the Rust ``CompiledLens``."""
    import optiland_small  # noqa: PLC0415

    compiled = optiland_small.CompiledLens(
        snap.kinds,
        snap.radius,
        snap.conic,
        snap.thickness,
        snap.index,
        snap.first_z,
        snap.stop,
        snap.epd,
        snap.max_field,
        snap.primary,
        snap.num_rings,
    )
    compiled.set_operands(snap.f2_weight, snap.f2_target, snap.spots)
    compiled.set_variables(snap.variables)
    return compiled
