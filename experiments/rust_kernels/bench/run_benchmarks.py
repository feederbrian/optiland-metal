"""Benchmark Optiland's heaviest kernels against Numba and Rust ports.

Usage (from the repository root, after ``maturin develop --release``):

    .venv/bin/python experiments/rust_kernels/bench/run_benchmarks.py
        [--quick] [--only huygens,trace,fft,end_to_end] [--tag NAME]

Writes ``results/benchmarks[_NAME].json`` (render it with ``report.py``).
Every implementation runs on identical inputs, and its output is checked
against Optiland's before it is timed.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import common
import numba
import numba_controls
import numpy as np
import optiland_rs
import scipy
import scipy.fft

from optiland.psf import huygens_fresnel_strategies as hfs
from optiland.psf.huygens_fresnel import ScalarHuygensPSF

RESULTS = Path(__file__).resolve().parents[1] / "results"
THREADS = os.cpu_count() or 1
SECTIONS = ("huygens", "trace", "fft", "end_to_end")


def with_numba_threads(n, fn):
    """Wrap ``fn`` so it runs with ``n`` Numba threads."""

    def run():
        previous = numba.get_num_threads()
        numba.set_num_threads(n)
        try:
            return fn()
        finally:
            numba.set_num_threads(previous)

    return run


def timed(label, fn, repeats=5):
    """Best-of timing with a progress line."""
    seconds = common.best_of(fn, repeats=repeats)
    print(f"    {label:38s} {seconds * 1e3:10.2f} ms", flush=True)
    return seconds


# --------------------------------------------------------------------------
# Huygens-Fresnel summation
# --------------------------------------------------------------------------


def huygens_case(lens, num_rays, image_size):
    """Time every Huygens implementation on one captured Optiland input."""
    image, pupil, wl, rp, center = common.capture_huygens_inputs(
        lens, num_rays, image_size
    )
    pairs = image[0].size * pupil[0].size
    print(f"\n  Huygens PSF: {image[0].shape} image x {pupil[0].size} pupil points")
    reference = common.huygens_optiland(image, pupil, wl, rp, center)
    phase = numba_controls.max_phase(image, pupil[:3], wl)
    assert phase < numba_controls.FAST_SINCOS_LIMIT

    def rust(variant, parallel=True):
        return lambda: optiland_rs.huygens_psf(
            *image, *pupil, wl, rp, center, variant=variant, parallel=parallel
        )

    impls = {
        "optiland_numba": lambda: common.huygens_optiland(image, pupil, wl, rp, center),
        "numba_hoisted": lambda: numba_controls.huygens_hoisted(
            *image, *pupil, wl, rp, center
        ),
        "numba_simd": lambda: numba_controls.huygens_simd(
            *image, *pupil, wl, rp, center
        ),
        "rust_literal": rust("literal"),
        "rust_hoisted": rust("hoisted"),
        "rust_simd": rust("simd"),
    }
    for name, fn in impls.items():
        assert np.abs(fn() - reference).max() < 1e-9 * reference.max(), name

    timings = {}
    for name, fn in impls.items():
        timings[f"{name}@{THREADS}t"] = timed(
            f"{name} ({THREADS} threads)", with_numba_threads(THREADS, fn)
        )
    # Per-core comparison: everything on one thread.
    for name in ("optiland_numba", "numba_hoisted", "numba_simd"):
        timings[f"{name}@1t"] = timed(
            f"{name} (1 thread)", with_numba_threads(1, impls[name]), repeats=3
        )
    for variant in ("literal", "simd"):
        timings[f"rust_{variant}@1t"] = timed(
            f"rust_{variant} (1 thread)", rust(variant, parallel=False), repeats=3
        )
    return {
        "image": list(image[0].shape),
        "pupil_points": int(pupil[0].size),
        "pairs": int(pairs),
        "max_phase_rad": phase,
        "seconds": timings,
    }


def bench_huygens(cases):
    """Huygens-Fresnel PSF summation on the Double Gauss, field (0, 1)."""
    lens = common.LENSES["double_gauss"]()
    return [huygens_case(lens, num_rays, size) for num_rays, size in cases]


# --------------------------------------------------------------------------
# Sequential trace
# --------------------------------------------------------------------------


def trace_case(lens_name, lens, table, per_axis):
    """Time the surface loop of one lens at one ray count."""
    rays = common.launch_rays(lens, per_axis)
    columns = common.ray_columns(rays)
    n = columns[0].size
    print(f"\n  Trace {lens_name}: {n:,} rays x {len(table.kinds)} surfaces")
    reference = common.trace_optiland(lens, rays)

    def rust(parallel):
        return lambda: optiland_rs.trace_sequential(
            columns,
            *table.args(),
            tol=table.tol,
            max_iter=table.max_iter,
            parallel=parallel,
        )

    def numba_fused():
        return numba_controls.trace_sequential(
            *columns, *table.args(), table.tol, table.max_iter
        )

    for fn in (rust(True), numba_fused):
        for ref, got in zip(reference, fn(), strict=True):
            np.testing.assert_allclose(got, ref, rtol=0, atol=1e-9, equal_nan=True)

    timings = {
        "optiland_numpy@1t": timed(
            "optiland numpy surface loop",
            lambda: common.trace_optiland(lens, rays),
            repeats=3,
        ),
        "numba_fused@1t": timed(
            "numba fused (1 thread)", with_numba_threads(1, numba_fused)
        ),
        f"numba_fused@{THREADS}t": timed(
            f"numba fused ({THREADS} threads)",
            with_numba_threads(THREADS, numba_fused),
        ),
        "rust_fused@1t": timed("rust fused (1 thread)", rust(False)),
        f"rust_fused@{THREADS}t": timed(f"rust fused ({THREADS} threads)", rust(True)),
    }
    return {
        "lens": lens_name,
        "rays": int(n),
        "surfaces": len(table.kinds),
        "seconds": timings,
    }


def bench_trace(per_axis_list):
    """Surface loop for every benchmark lens and ray count."""
    rows = []
    for lens_name, make in common.LENSES.items():
        lens = make()
        table = common.surface_table(lens, lens.primary_wavelength)
        rows += [trace_case(lens_name, lens, table, n) for n in per_axis_list]
    return rows


# --------------------------------------------------------------------------
# FFT
# --------------------------------------------------------------------------


def fft_case(size, rng):
    """Time 2-D FFT implementations on one random complex array."""
    data = rng.standard_normal((size, size)) + 1j * rng.standard_normal((size, size))
    print(f"\n  FFT2 {size}x{size} complex128")
    reference = np.fft.fft2(data)
    assert np.abs(optiland_rs.fft2(data) - reference).max() < 1e-9 * size
    timings = {
        "numpy_pocketfft@1t": timed(
            "numpy.fft.fft2 (Optiland's path)", lambda: np.fft.fft2(data)
        ),
        "scipy_fft@1t": timed(
            "scipy.fft.fft2 workers=1", lambda: scipy.fft.fft2(data, workers=1)
        ),
        f"scipy_fft@{THREADS}t": timed(
            f"scipy.fft.fft2 workers={THREADS}",
            lambda: scipy.fft.fft2(data, workers=THREADS),
        ),
        "rustfft@1t": timed(
            "rustfft (1 thread)", lambda: optiland_rs.fft2(data, parallel=False)
        ),
        f"rustfft@{THREADS}t": timed(
            f"rustfft ({THREADS} threads)", lambda: optiland_rs.fft2(data)
        ),
    }
    return {"size": size, "seconds": timings}


def bench_fft(sizes):
    """2-D complex FFTs of increasing size."""
    rng = np.random.default_rng(0)
    return [fft_case(size, rng) for size in sizes]


# --------------------------------------------------------------------------
# End to end: swap a kernel inside a real Optiland call
# --------------------------------------------------------------------------


def _rust_huygens_kernel(*args):
    """Drop-in for ``NumbaSummation._huygens_fresnel_summation``."""
    image = (np.ascontiguousarray(a, np.float64) for a in args[:3])
    pupil = (np.ascontiguousarray(a, np.float64).ravel() for a in args[3:8])
    center = tuple(float(c) for c in np.ravel(args[10]))
    return optiland_rs.huygens_psf(
        *image, *pupil, float(args[8]), float(args[9]), center
    )


def huygens_end_to_end(num_rays, image_size):
    """A full ``ScalarHuygensPSF`` with Optiland's kernel, then with Rust's."""
    lens = common.LENSES["double_gauss"]()
    wl = lens.primary_wavelength

    def analysis():
        ScalarHuygensPSF(lens, (0, 1), wl, num_rays=num_rays, image_size=image_size)

    print(f"\n  End to end: ScalarHuygensPSF({num_rays} rays, {image_size}px)")
    stock = timed("optiland as shipped", analysis, repeats=3)
    original = hfs.NumbaSummation._huygens_fresnel_summation
    hfs.NumbaSummation._huygens_fresnel_summation = staticmethod(_rust_huygens_kernel)
    try:
        swapped = timed("with rust_simd kernel swapped in", analysis, repeats=3)
    finally:
        hfs.NumbaSummation._huygens_fresnel_summation = original
    return {
        "workload": f"ScalarHuygensPSF, {num_rays} rays, {image_size}px image",
        "stock": stock,
        "rust": swapped,
    }


def trace_end_to_end(lens_name, per_axis):
    """A full ``lens.trace(record=False)``, then with the Rust surface loop."""
    lens = common.LENSES[lens_name]()
    wl = lens.primary_wavelength
    table = common.surface_table(lens, wl)
    group = lens.surfaces

    def rust_loop(rays, skip=0, record=True):
        assert skip == 0 and not record
        out = optiland_rs.trace_sequential(
            common.ray_columns(rays),
            *table.args(),
            tol=table.tol,
            max_iter=table.max_iter,
        )
        for name, column in zip(common.RAY_FIELDS, out, strict=True):
            setattr(rays, name, column)
        return rays

    def full_trace():
        return lens.trace(0, 1, wl, per_axis, "uniform", record=False)

    print(f"\n  End to end: {lens_name}.trace(uniform {per_axis}^2, record=False)")
    stock = timed("optiland as shipped", full_trace, repeats=3)
    expected = full_trace()
    group.trace = rust_loop  # instance attribute shadows SurfaceGroup.trace
    try:
        np.testing.assert_allclose(full_trace().x, expected.x, rtol=0, atol=1e-9)
        swapped = timed("with rust surface loop swapped in", full_trace, repeats=3)
    finally:
        del group.trace
    return {
        "workload": f"{lens_name}.trace, {expected.x.size:,} rays",
        "stock": stock,
        "rust": swapped,
    }


def bench_end_to_end(huygens, trace_per_axis):
    """Kernel swaps inside real Optiland entry points."""
    return [
        huygens_end_to_end(*huygens),
        trace_end_to_end("double_gauss", trace_per_axis),
        trace_end_to_end("aspheric_double_gauss", trace_per_axis),
    ]


# --------------------------------------------------------------------------


def environment():
    """Hardware and library versions recorded with the results."""
    cpu = "unknown"
    try:
        with open("/proc/cpuinfo") as f:
            cpu = next(
                line.split(":", 1)[1].strip()
                for line in f
                if line.startswith("model name")
            )
    except (OSError, StopIteration):
        pass
    rustc = subprocess.run(
        ["rustc", "--version"], capture_output=True, text=True, check=False
    ).stdout.strip()
    return {
        "cpu": cpu,
        "threads": THREADS,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "numba": numba.__version__,
        "numba_svml": bool(numba.config.USING_SVML),
        "rustc": rustc,
        "date": time.strftime("%Y-%m-%d"),
    }


def main():
    """Run the selected sections and write the JSON results."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="small smoke-run sizes")
    parser.add_argument("--only", default=",".join(SECTIONS), help="sections to run")
    parser.add_argument("--tag", default="", help="suffix for the output file")
    args = parser.parse_args()

    if args.quick:
        sizes = {
            "huygens": ([(48, 32)],),
            "trace": ([64, 160],),
            "fft": ([256, 512],),
            "end_to_end": ((48, 32), 160),
        }
    else:
        sizes = {
            "huygens": ([(64, 64), (128, 128)],),
            "trace": ([36, 113, 357, 1129],),  # ~1e3, 1e4, 1e5, 1e6 rays
            "fft": ([256, 512, 1024, 2048],),
            "end_to_end": ((128, 128), 512),
        }
    runners = {
        "huygens": bench_huygens,
        "trace": bench_trace,
        "fft": bench_fft,
        "end_to_end": bench_end_to_end,
    }

    results = {"environment": environment()}
    print("Environment:", results["environment"])
    for section in args.only.split(","):
        print(f"\n== {section} ==")
        results[section] = runners[section](*sizes[section])

    RESULTS.mkdir(exist_ok=True)
    tag = args.tag or ("quick" if args.quick else "")
    path = RESULTS / (f"benchmarks_{tag}.json" if tag else "benchmarks.json")
    path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
