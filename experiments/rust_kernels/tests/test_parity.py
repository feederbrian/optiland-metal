"""Every compiled kernel must reproduce Optiland's own results.

Run from the repository root after ``maturin develop --release``:

    .venv/bin/python -m pytest experiments/rust_kernels/tests -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))

import common  # noqa: E402
import numba_controls  # noqa: E402

optiland_rs = pytest.importorskip("optiland_rs")


@pytest.fixture(scope="module", params=list(common.LENSES))
def traced(request):
    lens = common.LENSES[request.param]()
    table = common.surface_table(lens, lens.primary_wavelength)
    rays = common.launch_rays(lens, 96)
    return table, common.ray_columns(rays), common.trace_optiland(lens, rays)


def _assert_rays_match(reference, result):
    for name, ref, got in zip(common.RAY_FIELDS, reference, result):
        assert np.array_equal(np.isnan(ref), np.isnan(got)), name
        # Positions and OPD in mm; Newton stops at a 1e-10 mm residual, and
        # the even-asphere polynomial uses Horner's rule instead of powers.
        atol = 1e-9 if name in ("x", "y", "z", "opd") else 1e-12
        np.testing.assert_allclose(got, ref, rtol=0, atol=atol, equal_nan=True, err_msg=name)


@pytest.mark.parametrize("parallel", [True, False])
def test_rust_trace_matches_optiland(traced, parallel):
    table, columns, reference = traced
    result = optiland_rs.trace_sequential(
        columns, *table.args(), tol=table.tol, max_iter=table.max_iter, parallel=parallel
    )
    _assert_rays_match(reference, result)


def test_numba_trace_matches_optiland(traced):
    table, columns, reference = traced
    result = numba_controls.trace_sequential(*columns, *table.args(), table.tol, table.max_iter)
    _assert_rays_match(reference, result)


def test_spherical_trace_is_bit_identical():
    """Without Newton iterations the fused loop does Optiland's exact arithmetic."""
    lens = common.LENSES["double_gauss"]()
    table = common.surface_table(lens, lens.primary_wavelength)
    rays = common.launch_rays(lens, 64)
    reference = common.trace_optiland(lens, rays)
    result = optiland_rs.trace_sequential(common.ray_columns(rays), *table.args())
    for name, ref, got in zip(common.RAY_FIELDS, reference, result):
        if name != "i":  # exp() of the absorption term may differ by 1 ulp
            assert np.array_equal(ref, got), name


@pytest.fixture(scope="module")
def huygens_case():
    lens = common.LENSES["double_gauss"]()
    image, pupil, wavelength, rp = common.capture_huygens_inputs(lens, num_rays=48, image_size=24)
    return image, pupil, wavelength, rp, common.huygens_optiland(image, pupil, wavelength, rp)


@pytest.mark.parametrize("variant", ["literal", "hoisted", "simd"])
@pytest.mark.parametrize("parallel", [True, False])
def test_rust_huygens_matches_optiland(huygens_case, variant, parallel):
    image, pupil, wavelength, rp, reference = huygens_case
    result = optiland_rs.huygens_psf(
        *image, *pupil, wavelength, rp, variant=variant, parallel=parallel
    )
    # The reference is compiled with fastmath, so summation order differs.
    np.testing.assert_allclose(result, reference, rtol=0, atol=1e-9 * reference.max())


def test_numba_hoisted_huygens_matches_optiland(huygens_case):
    image, pupil, wavelength, rp, reference = huygens_case
    result = numba_controls.huygens_hoisted(*image, *pupil, wavelength, rp)
    np.testing.assert_allclose(result, reference, rtol=0, atol=1e-9 * reference.max())


@pytest.mark.parametrize("shape", [(64, 64), (96, 40), (128, 256)])
@pytest.mark.parametrize("parallel", [True, False])
def test_rust_fft2_matches_numpy(shape, parallel):
    rng = np.random.default_rng(0)
    data = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    result = optiland_rs.fft2(data, parallel=parallel)
    np.testing.assert_allclose(result, np.fft.fft2(data), rtol=0, atol=1e-10 * np.sqrt(data.size))


def test_numba_simd_huygens_matches_optiland(huygens_case):
    image, pupil, wavelength, rp, reference = huygens_case
    # The Numba reduction (no FMA) is exact only below this phase.
    assert numba_controls.max_phase(image, pupil[:3], wavelength) < numba_controls.FAST_SINCOS_LIMIT
    result = numba_controls.huygens_simd(*image, *pupil, wavelength, rp)
    np.testing.assert_allclose(result, reference, rtol=0, atol=1e-9 * reference.max())


def test_mismatched_inputs_raise_instead_of_crashing():
    a, b = np.zeros((4, 4)), np.zeros(3)
    with pytest.raises(ValueError, match="pupil"):
        optiland_rs.huygens_psf(a, a, a, b, b, b, b, np.zeros(2), 5e-4, 100.0)
    with pytest.raises(ValueError, match="image"):
        optiland_rs.huygens_psf(a, a, np.zeros((4, 3)), b, b, b, b, b, 5e-4, 100.0)
    rays = [np.zeros(2)] * 8
    one = np.zeros(1)
    with pytest.raises(ValueError, match="surface"):
        optiland_rs.trace_sequential(
            rays, np.zeros(2, np.int64), one, one, one, np.zeros((2, 1)), one, one, one
        )
