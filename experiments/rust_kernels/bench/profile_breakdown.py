"""Where does each Optiland analysis spend its time?

For each workload, splits cProfile time into the kernels a compiled port can
replace (surface ray-trace loop, Huygens summation, FFT) and everything else
(Python orchestration: paraxial setup, material caches, backend dispatch,
array bookkeeping). This bounds what any Rust port could achieve end to end.

    .venv/bin/python experiments/rust_kernels/bench/profile_breakdown.py
"""

from __future__ import annotations

import cProfile
import json
import pstats
import time
from pathlib import Path

import common

from optiland.analysis import SpotDiagram
from optiland.mtf import FFTMTF, GeometricMTF
from optiland.psf import FFTPSF, HuygensPSF
from optiland.wavefront import OPD

RESULTS = Path(__file__).resolve().parents[1] / "results"

# (file suffix, function name, required caller file suffix or None) -> bucket.
# The surface loop also serves the paraxial tracer (one-ray Python traces the
# Rust port does not cover), so only time called from the real tracer counts.
KERNELS = {
    (
        "surfaces/surface_group.py",
        "trace",
        "raytrace/real_ray_tracer.py",
    ): "real-ray surface loop",
    ("psf/huygens_fresnel_strategies.py", "compute", None): "Huygens summation",
    ("numpy/fft/_pocketfft.py", "_raw_fft", None): "FFT",
}


def breakdown(fn) -> dict:
    """Wall time of ``fn`` and the share of profiled time in each bucket."""
    fn()
    start = time.perf_counter()
    fn()
    wall = time.perf_counter() - start
    profile = cProfile.Profile()
    profile.enable()
    fn()
    profile.disable()
    stats = pstats.Stats(profile).stats
    total = max(ct for (_, _, _), (_, _, _, ct, _) in stats.items())
    buckets = dict.fromkeys(KERNELS.values(), 0.0)
    for (filename, _, name), (_, _, _, cumtime, callers) in stats.items():
        for (suffix, func, caller_suffix), bucket in KERNELS.items():
            if name != func or not filename.endswith(suffix):
                continue
            if caller_suffix is None:
                buckets[bucket] += cumtime
            else:
                buckets[bucket] += sum(
                    ct
                    for caller, (_, _, _, ct) in callers.items()
                    if caller[0].endswith(caller_suffix)
                )
    shares = {k: v / total for k, v in buckets.items()}
    shares["Python orchestration + other"] = max(0.0, 1.0 - sum(shares.values()))
    return {"wall_seconds": wall, "shares": shares}


def main():
    """Profile each analysis and write ``results/profile_breakdown.json``."""
    dg = common.LENSES["double_gauss"]()
    wl = dg.primary_wavelength
    workloads = {
        "lens.trace, 205k rays": lambda: dg.trace(0, 1, wl, 512, "uniform"),
        "SpotDiagram (3 fields x 3 wl)": lambda: SpotDiagram(dg, num_rings=30),
        "OPD map": lambda: OPD(dg, (0, 1), wl, num_rings=64),
        "GeometricMTF": lambda: GeometricMTF(dg),
        "FFTPSF 256 rays, 1024 grid": lambda: FFTPSF(
            dg, (0, 1), wl, num_rays=256, grid_size=1024
        ),
        "FFTMTF 256 rays, 1024 grid": lambda: FFTMTF(dg, num_rays=256, grid_size=1024),
        "HuygensPSF 128 rays, 128 px": lambda: HuygensPSF(
            dg, (0, 1), wl, num_rays=128, image_size=128
        ),
    }
    results = {}
    for name, fn in workloads.items():
        results[name] = breakdown(fn)
        shares = ", ".join(
            f"{k} {v:.0%}" for k, v in results[name]["shares"].items() if v >= 0.005
        )
        print(
            f"{name:32s} {results[name]['wall_seconds'] * 1e3:8.0f} ms   {shares}",
            flush=True,
        )
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "profile_breakdown.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
