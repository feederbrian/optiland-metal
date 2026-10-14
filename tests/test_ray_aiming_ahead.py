"""Robust ray aiming: work done once per system state or ahead of request.

The robust aimer solves the stop radius once per exact system state and,
when the system changes, batches the calibrations and polishes of several
fields (``solve_ahead``). None of it may change a single bit of a result.
Each test replays one sequence of system changes and requests on two copies
of a lens, one with ``solve_ahead=False``, and requires identical rays and
identical aimer reports.
"""

from __future__ import annotations

import numpy as np
import pytest

import optiland.backend as be
from optiland.rays.ray_aiming.initialization import RealReferenceStrategy
from optiland.rays.ray_aiming.robust import RobustRayAimer, _system_state_key
from optiland.samples.objectives import (
    CookeTriplet,
    ProjectionLens120FOV,
    WideAngle100FOV,
)
from optiland.samples.simple import AsphericSinglet

RAY_FIELDS = ("x", "y", "z", "L", "M", "N", "i", "opd")


def _finite_triplet():
    lens = CookeTriplet()
    lens.updater.set_thickness(300.0, 0)
    lens.fields.set_type("object_height")
    return lens


LENSES = {
    "cooke_triplet": CookeTriplet,
    "finite_triplet": _finite_triplet,
    "wide_angle_100": WideAngle100FOV,
    "projection_120": ProjectionLens120FOV,
    "aspheric_singlet": AsphericSinglet,
}


def _robust_aimer(lens) -> RobustRayAimer:
    aimer = lens.ray_tracer.ray_generator.aimer
    return getattr(aimer, "wrapped_aimer", aimer)


def _steps(lens):
    """System changes like an optimizer's: steps, restores, an aperture change."""
    stop = lens.surfaces.stop_index
    pre = 1
    post = min(stop + 1, len(lens.surfaces.surfaces) - 2)
    r_pre = float(lens.surfaces[pre].geometry.radius)
    t_post = float(be.to_numpy(lens.surfaces[post].thickness))
    aperture = lens.aperture.value

    def nothing():
        pass

    def step_pre():
        lens.updater.set_radius(r_pre * (1 + 1e-5), pre)

    def restore_pre():
        lens.updater.set_radius(r_pre, pre)

    def step_post():
        lens.updater.set_thickness(t_post * (1 + 1e-5), post)

    def restore_post():
        lens.updater.set_thickness(t_post, post)

    def widen():
        lens.set_aperture(lens.aperture.ap_type, aperture * 1.1)

    def restore_aperture():
        lens.set_aperture(lens.aperture.ap_type, aperture)

    return [
        nothing,
        step_pre,
        restore_pre,
        step_post,
        restore_post,
        step_pre,
        restore_pre,
        widen,
        restore_aperture,
        nothing,
    ]


def _replay(lens, cache: bool, solve_ahead: bool) -> tuple[list, list, dict]:
    lens.ray_tracer.set_aiming(
        "robust", max_iter=30, tol=1e-9, cache=cache, solve_ahead=solve_ahead
    )
    wavelengths = [w.value for w in lens.wavelengths.wavelengths]
    rays_out, reports = [], []
    counts = {"calibrations": 0, "ahead_polishes": 0}
    instrumented = set()
    for step in _steps(lens):
        step()
        for hy in (0.0, 0.7, 1.0):
            for wl in wavelengths:
                for generic in (False, True):
                    if generic:
                        rays = lens.trace_generic(0.0, hy, 0.0, 0.0, wl)
                    else:
                        rays = lens.trace(0.0, hy, wl, 4, "hexapolar", record=False)
                    aimer = _robust_aimer(lens)
                    if id(aimer) not in instrumented:
                        instrumented.add(id(aimer))
                        _count_calls(aimer, counts)
                    rays_out.append(
                        np.stack(
                            [
                                np.ravel(be.to_numpy(getattr(rays, f)))
                                for f in RAY_FIELDS
                            ]
                        )
                    )
                    reports.append(aimer.last_report)
    return rays_out, reports, counts


def _count_calls(aimer, counts):
    calibrate = aimer._calibrate_field
    take = aimer._take_polish

    def counted_calibrate(*args, **kwargs):
        counts["calibrations"] += 1
        return calibrate(*args, **kwargs)

    def counted_take(*args, **kwargs):
        result = take(*args, **kwargs)
        counts["ahead_polishes"] += result is not None
        return result

    aimer._calibrate_field = counted_calibrate
    aimer._take_polish = counted_take


@pytest.mark.parametrize("cache", [False, True])
@pytest.mark.parametrize("name", sorted(LENSES))
def test_solve_ahead_is_bit_identical(name, cache, set_test_backend):
    if be.get_backend() != "numpy":
        pytest.skip("solve_ahead runs on the NumPy float64 backend")
    rays_ahead, reports_ahead, counts_ahead = _replay(
        LENSES[name](), cache, solve_ahead=True
    )
    rays_plain, reports_plain, counts_plain = _replay(
        LENSES[name](), cache, solve_ahead=False
    )
    assert len(rays_ahead) == len(rays_plain)
    for i, (a, b) in enumerate(zip(rays_ahead, rays_plain, strict=True)):
        assert a.shape == b.shape, i
        assert np.array_equal(a, b, equal_nan=True), i
    assert reports_ahead == reports_plain
    # The batched paths were exercised, not bypassed.
    assert counts_plain["ahead_polishes"] == 0
    assert counts_ahead["calibrations"] < counts_plain["calibrations"]
    assert counts_ahead["ahead_polishes"] > 0


def test_stop_radius_follows_every_change(set_test_backend):
    """The memoized stop radius equals a fresh solve after every change."""
    lens = WideAngle100FOV()  # image F-number: the EPD depends on every surface
    lens.ray_tracer.set_aiming("robust", max_iter=30, tol=1e-9)
    lens.trace(0.0, 1.0, 0.5876, 3, "hexapolar")
    aimer = _robust_aimer(lens)
    last = len(lens.surfaces.surfaces) - 2
    t_last = float(be.to_numpy(lens.surfaces[last].thickness))
    r_first = float(lens.surfaces[1].geometry.radius)
    changes = [
        lambda: None,
        lambda: lens.updater.set_thickness(t_last * (1 + 1e-9), last),
        lambda: lens.updater.set_thickness(t_last, last),
        lambda: lens.updater.set_radius(r_first * (1 - 1e-7), 1),
        lambda: lens.set_aperture("imageFNO", 4.4),
    ]
    for change in changes:
        change()
        memo = aimer._stop_radius(_system_state_key(lens))
        assert memo == RealReferenceStrategy(lens).calculate_stop_radius()


def _asphere_before_stop():
    from optiland.optic import Optic

    lens = Optic()
    lens.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    lens.surfaces.add(
        index=1,
        radius=20.0,
        thickness=5.0,
        material="N-BK7",
        surface_type="even_asphere",
        coefficients=[-2.0e-4, -5.0e-6],
    )
    lens.surfaces.add(index=2, radius=-60.0, thickness=3.0)
    lens.surfaces.add(index=3, radius=be.inf, thickness=30.0, is_stop=True)
    lens.surfaces.add(index=4)
    lens.set_aperture(aperture_type="EPD", value=12.0)
    lens.fields.set_type(field_type="angle")
    lens.fields.add(y=0)
    lens.fields.add(y=5)
    lens.wavelengths.add(value=0.55)
    return lens


def test_stop_radius_sees_in_place_asphere_writes(set_test_backend):
    """An in-place coefficient write (as the optimizer makes) is a new state."""
    lens = _asphere_before_stop()
    lens.ray_tracer.set_aiming("robust", max_iter=30, tol=1e-9)
    lens.trace(0.0, 1.0, 0.55, 3, "hexapolar")
    aimer = _robust_aimer(lens)
    first = aimer._stop_radius(_system_state_key(lens))
    assert first == RealReferenceStrategy(lens).calculate_stop_radius()
    lens.updater.set_asphere_coeff(-3.0e-4, 1, 0)
    second = aimer._stop_radius(_system_state_key(lens))
    assert second == RealReferenceStrategy(lens).calculate_stop_radius()
    assert second != first
