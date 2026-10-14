"""Every fast path must reproduce Optiland's own numbers.

Run from the repository root after ``maturin develop --release``:

    .venv/bin/python -m pytest experiments/rust_small_ops/tests -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))

import workload as common  # noqa: E402
import lean  # noqa: E402
import numba_lens  # noqa: E402
import snapshot  # noqa: E402

from optiland import optimization  # noqa: E402
from optiland.optimization.operand.ray import RayOperand  # noqa: E402
from optiland.surfaces.standard_surface import Surface  # noqa: E402

optiland_small = pytest.importorskip("optiland_small")
patches = pytest.importorskip("patches")

RTOL = 1e-10


@pytest.fixture(scope="module")
def case():
    lens = common.tutorial_triplet()
    problem = common.tutorial_problem(lens)
    snap = snapshot.capture(lens, problem)
    x0 = np.array([float(v.value) for v in problem.variables])
    return {
        "lens": lens,
        "problem": problem,
        "optimizer": optimization.OptimizerGeneric(problem),
        "x0": x0,
        "xs": [x0 + d * np.linspace(-1, 1, x0.size) for d in (0.0, 0.05, -0.2)],
        "impls": {
            "rust": snapshot.to_rust(snap),
            "numba": numba_lens.NumbaLens(snap),
            "python": lean.PythonLens(snap),
            "numpy": lean.NumpyLens(snap),
        },
    }


def _reset(case):
    case["optimizer"]._fun(case["x0"])


def test_first_order_matches(case):
    _reset(case)
    lens = case["lens"]
    f2, epl = float(lens.paraxial.f2()), float(lens.paraxial.EPL())
    for name, impl in case["impls"].items():
        assert impl.f2() == pytest.approx(f2, rel=RTOL), name
    for name in ("rust", "python"):
        impl = case["impls"][name]
        assert impl.epl() == pytest.approx(epl, rel=RTOL), name


@pytest.mark.parametrize("w", [0, 1, 2])
@pytest.mark.parametrize("ray", [(0, 0, 0, 0), (0, 1, 0.3, 0.8), (0.4, 0.7, -0.9, 0.1)])
def test_single_ray_matches(case, w, ray):
    _reset(case)
    lens = case["lens"]
    wl = lens.wavelengths.get_wavelengths()[w]
    ref = lens.trace_generic(*ray, wl)
    expected = [float(np.ravel(getattr(ref, a))[0]) for a in "xyzLMN"]
    for name in ("rust", "numba", "python"):
        got = case["impls"][name].trace_generic(*map(float, ray), w)
        np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12, err_msg=name)


@pytest.mark.parametrize("hy", common.FIELDS)
def test_rms_spot_matches(case, hy):
    _reset(case)
    expected = float(
        RayOperand.rms_spot_size(case["lens"], -1, 0, hy, common.NUM_RINGS, "all")
    )
    impls = case["impls"]
    assert impls["rust"].rms_spot(0.0, hy) == pytest.approx(expected, rel=RTOL)
    assert impls["numba"].rms_spot(0.0, hy) == pytest.approx(expected, rel=RTOL)
    for name in ("python", "numpy"):
        impl = impls[name]
        assert impl.rms_spot(impl.epl(), 0.0, hy) == pytest.approx(expected, rel=RTOL)


def test_merit_matches_at_several_designs(case):
    for x in case["xs"]:
        expected = float(case["optimizer"]._fun(x))
        impls = case["impls"]
        assert impls["rust"].merit_at(list(x)) == pytest.approx(expected, rel=RTOL)
        for name in ("numba", "python", "numpy"):
            assert impls[name].merit_at(x) == pytest.approx(expected, rel=RTOL), name


def test_residuals_match(case):
    x = case["xs"][1]
    case["optimizer"]._fun(x)
    expected = [float(op.fun()) for op in case["problem"].operands]
    got = case["impls"]["rust"].residuals_at(list(x))
    np.testing.assert_allclose(got, expected, rtol=RTOL)


def test_batched_evaluations_match_single(case):
    rows = np.array(case["xs"])
    rust, nb = case["impls"]["rust"], case["impls"]["numba"]
    single = [rust.merit_at(list(r)) for r in rows]
    np.testing.assert_array_equal(rust.merit_many(rows, parallel=True), single)
    np.testing.assert_array_equal(rust.merit_many(rows, parallel=False), single)
    np.testing.assert_allclose(nb.merit_many(rows), single, rtol=RTOL)
    res = rust.residuals_many(rows)
    for row, r in zip(res, rows, strict=True):
        np.testing.assert_array_equal(row, rust.residuals_at(list(r)))


@pytest.mark.parametrize(
    "names",
    [
        ("rust_surface_leaf",),
        ("evaluation_scope",),
        ("direct_backend",),
        ("evaluation_scope", "direct_backend", "rust_surface_leaf"),
    ],
)
def test_patches_preserve_results_and_restore(case, names):
    import contextlib  # noqa: PLC0415

    lens, opt = case["lens"], case["optimizer"]
    wl = lens.primary_wavelength
    merits = [float(opt._fun(x)) for x in case["xs"]]
    ray = lens.trace_generic(0.0, 1.0, 0.3, 0.8, wl)
    before = Surface.__dict__["_trace_real"]
    with contextlib.ExitStack() as stack:
        for name in names:
            stack.enter_context(getattr(patches, name)())
        # The lens changes between calls: a stale cache would show here.
        assert [float(opt._fun(x)) for x in case["xs"]] == merits
        patched = lens.trace_generic(0.0, 1.0, 0.3, 0.8, wl)
    for attr in "xyzLMN":
        assert np.array_equal(getattr(patched, attr), getattr(ray, attr)), attr
    assert Surface.__dict__["_trace_real"] is before
    import optiland.backend as be  # noqa: PLC0415

    assert "sqrt" not in vars(be)


def test_mismatched_inputs_raise():
    with pytest.raises(ValueError, match="equal length"):
        optiland_small.trace_surface(
            [np.zeros(2)] * 7 + [np.zeros(3)], 1, 10.0, 0.0, 1.0, 1.5, 0.0
        )
    with pytest.raises(ValueError, match="surface kind"):
        optiland_small.trace_surface([np.zeros(1)] * 8, 7, 10.0, 0.0, 1.0, 1.5, 0.0)
    with pytest.raises(ValueError, match="length"):
        optiland_small.CompiledLens([0, 1], [1.0], [0.0], [1.0], [[1.0]], 0, 1, 1, 1, 0, 1)
