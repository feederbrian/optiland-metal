"""Load a ``uw1-optic/1`` export (the pipeline's lens format) into Optiland.

The lens data itself is not part of this repository. Point the harness at an
export with ``--optic PATH`` or the ``UW2_OPTIC`` environment variable.

Every load is checked against the export's own records: surface vertex
positions (``z``) and refractive indices at the five sampled wavelengths
(``n_ref``), so a catalogue or version difference cannot pass silently.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np

import optiland.backend as be
from optiland import optic as optic_module
from optiland.materials import IdealMaterial, Material
from optiland.physical_apertures import RadialAperture

DEFAULT_OPTIC = Path("/home/user/test-repo/uw2A_X14gs_dual28n3_s1_V2.optic.json")
INDEX_RTOL = 1e-9
Z_ATOL = 1e-9


def optic_path(path: str | os.PathLike | None = None) -> Path:
    """Resolve the export path: argument, ``UW2_OPTIC``, then the default."""
    chosen = path or os.environ.get("UW2_OPTIC") or DEFAULT_OPTIC
    resolved = Path(chosen)
    if not resolved.exists():
        raise FileNotFoundError(
            f"{resolved} not found. Pass --optic PATH or set UW2_OPTIC to a "
            "uw1-optic/1 export."
        )
    return resolved


def _num(value) -> float:
    if isinstance(value, str):
        return {"inf": math.inf, "-inf": -math.inf}[value]
    return float(value)


def _material(record: dict):
    kind = record.get("kind")
    if kind == "ideal":
        return IdealMaterial(n=float(record["index"]))
    if kind == "catalogue":
        if record.get("reference"):
            return Material(record["name"], reference=record["reference"])
        return Material(record["name"])
    raise NotImplementedError(f"material kind {kind!r}")


def load(path: str | os.PathLike | None = None, verify: bool = True):
    """Build the Optic described by a ``uw1-optic/1`` export."""
    data = json.loads(optic_path(path).read_text())
    if data.get("format") != "uw1-optic/1":
        raise ValueError(f"unsupported format {data.get('format')!r}")

    lens = optic_module.Optic()
    for s in data["surfaces"]:
        cs = s["cs"]
        if any(float(cs[k]) for k in ("x", "y", "rx", "ry", "rz")):
            raise NotImplementedError("decentered or tilted surface")
        if s.get("coating") or s.get("bsdf") or s.get("is_reflective"):
            raise NotImplementedError("coating, BSDF or mirror")
        kwargs = {
            "index": s["index"],
            "radius": _num(s["radius"]),
            "thickness": _num(s["thickness"]),
            "conic": _num(s["conic"]),
            "is_stop": bool(s["is_stop"]),
            "comment": s.get("comment", ""),
            "material": _material(s["material"]),
        }
        geometry = s["geometry"]
        if geometry == "EvenAsphere":
            kwargs.update(
                surface_type="even_asphere",
                coefficients=[float(c) for c in s["coefficients"]],
                tol=float(s["tol"]),
                max_iter=int(s["max_iter"]),
            )
        elif geometry not in ("StandardGeometry", "Plane"):
            raise NotImplementedError(f"geometry {geometry}")
        aperture = s.get("aperture")
        if aperture:
            if aperture.get("cls") != "RadialAperture":
                raise NotImplementedError(f"aperture {aperture.get('cls')}")
            kwargs["aperture"] = RadialAperture(
                r_max=float(aperture["r_max"]), r_min=float(aperture["r_min"])
            )
        lens.surfaces.add(**kwargs)

    ap = data["aperture"]
    lens.set_aperture(aperture_type=ap["type"], value=float(ap["value"]))
    lens.fields.set_type(field_type=data["field_type"])
    for f in data["fields"]:
        lens.fields.add(
            x=float(f["x"]), y=float(f["y"]), vx=float(f["vx"]), vy=float(f["vy"])
        )
    for w in data["wavelengths"]:
        if w.get("unit", "um") != "um":
            raise NotImplementedError("wavelength unit")
        lens.wavelengths.add(value=float(w["value"]), is_primary=bool(w["is_primary"]))
    aim = data.get("ray_aiming") or {}
    if aim:
        extra = {k: v for k, v in aim.items() if k not in ("mode", "max_iter", "tol")}
        lens.ray_tracer.set_aiming(
            mode=aim["mode"],
            max_iter=int(aim["max_iter"]),
            tol=float(aim["tol"]),
            **extra,
        )
    if data.get("pickups") or data.get("solves"):
        raise NotImplementedError("pickups / solves")
    if verify:
        check(lens, data)
    return lens


def check(lens, data: dict) -> dict:
    """Compare vertex positions and indices with the export's records."""
    positions = np.ravel(be.to_numpy(lens.surfaces.positions))
    worst_z = 0.0
    worst_n = 0.0
    for s, z in zip(data["surfaces"], positions, strict=True):
        expected = _num(s["z"])
        if math.isfinite(expected):
            worst_z = max(worst_z, abs(float(z) - expected))
        surface = lens.surfaces.surfaces[s["index"]]
        for wl, n_expected in s["material"].get("n_ref", {}).items():
            n = float(np.ravel(be.to_numpy(surface.material_post.n(float(wl))))[0])
            worst_n = max(worst_n, abs(n - n_expected) / n_expected)
    if worst_z > Z_ATOL:
        raise ValueError(f"vertex positions differ from the export by {worst_z:.3g} mm")
    if worst_n > INDEX_RTOL:
        raise ValueError(f"refractive indices differ from the export by {worst_n:.3g}")
    return {"max_z_error_mm": worst_z, "max_index_rel_error": worst_n}
