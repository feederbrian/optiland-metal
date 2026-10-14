"""Shared fixtures for the small-operations experiment.

The workload is the optimization case study from the Optiland docs
(``docs/examples/Tutorial_3d_Optimization_Case_Study_Cooke_Triplet.ipynb``):
a Cooke triplet, an ``f2`` target plus ``rms_spot_size`` at three fields over
all three wavelengths with ``num_rays=5`` (91-ray hexapolar traces), and seven
radius/thickness variables, solved by ``OptimizerGeneric``.
"""

from __future__ import annotations

import timeit
import warnings
from typing import TYPE_CHECKING

import numpy as np

import optiland.backend as be
from optiland import optic as optic_module
from optiland import optimization

if TYPE_CHECKING:
    from collections.abc import Callable

warnings.filterwarnings("ignore")
be.set_backend("numpy")

FIELDS = (0.0, 0.7, 1.0)
NUM_RINGS = 5
F2_TARGET = 50.0
SPOT_WEIGHT = 10.0


def tutorial_triplet() -> optic_module.Optic:
    """The tutorial's starting Cooke triplet (flat-ish +/-1000 mm radii)."""
    lens = optic_module.Optic()
    lens.surfaces.add(index=0, radius=np.inf, thickness=np.inf)
    lens.surfaces.add(index=1, radius=1000, thickness=4, material="SK16")
    lens.surfaces.add(index=2, radius=-1000, thickness=5)
    lens.surfaces.add(index=3, radius=-1000, thickness=4, material=("F2", "schott"))
    lens.surfaces.add(index=4, radius=1000, thickness=5, is_stop=True)
    lens.surfaces.add(index=5, radius=1000, thickness=4, material="SK16")
    lens.surfaces.add(index=6, radius=-1000, thickness=50)
    lens.surfaces.add(index=7)
    lens.set_aperture(aperture_type="EPD", value=10.0)
    lens.fields.set_type(field_type="angle")
    for y in (0, 14, 20):
        lens.fields.add(y=y)
    lens.wavelengths.add(value=0.4861)
    lens.wavelengths.add(value=0.5876, is_primary=True)
    lens.wavelengths.add(value=0.6563)
    return lens


# (variable type, surface number) in the order the tutorial adds them.
VARIABLES = (
    ("radius", 1),
    ("radius", 2),
    ("radius", 3),
    ("radius", 4),
    ("radius", 5),
    ("radius", 6),
    ("thickness", 6),
)


def tutorial_problem(lens, batching: bool = True):
    """The tutorial's second-stage merit function and variables."""
    problem = optimization.OptimizationProblem(batching=batching)
    problem.add_operand(
        operand_type="f2", target=F2_TARGET, weight=1, input_data={"optic": lens}
    )
    for hy in FIELDS:
        problem.add_operand(
            operand_type="rms_spot_size",
            target=0,
            weight=SPOT_WEIGHT,
            input_data={
                "optic": lens,
                "surface_number": -1,
                "Hx": 0,
                "Hy": hy,
                "wavelength": "all",
                "num_rays": NUM_RINGS,
            },
        )
    for kind, surface in VARIABLES:
        bounds = (0, 1000) if kind == "thickness" else (-1000, 1000)
        problem.add_variable(
            lens, kind, surface_number=surface, min_val=bounds[0], max_val=bounds[1]
        )
    return problem


def per_call(fn: Callable[[], object], repeats: int = 5) -> float:
    """Best seconds per call over ``repeats`` auto-ranged batches."""
    fn()
    timer = timeit.Timer(fn)
    number, _ = timer.autorange()
    return min(timer.repeat(repeat=repeats, number=number)) / number


def fmt(seconds: float) -> str:
    """Human-readable latency."""
    if seconds < 1e-6:
        return f"{seconds * 1e9:.0f} ns"
    if seconds < 1e-3:
        return f"{seconds * 1e6:.1f} us"
    return f"{seconds * 1e3:.2f} ms"
