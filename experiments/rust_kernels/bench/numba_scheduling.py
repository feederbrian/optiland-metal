"""Is Numba's weaker multi-thread scaling on big asphere traces a scheduling issue?

Newton-Raphson work varies across the pupil, so equal-sized static blocks of
rays are unequal amounts of work. Rayon balances this by work stealing.
``numba.set_parallel_chunksize`` hands out smaller chunks dynamically, which
should close the gap if scheduling is the cause.

    .venv/bin/python experiments/rust_kernels/bench/numba_scheduling.py
"""

from __future__ import annotations

import json
from pathlib import Path

import common
import numba
import numba_controls
import numpy as np
import optiland_rs
from run_benchmarks import THREADS, timed, with_numba_threads

RESULTS = Path(__file__).resolve().parents[1] / "results"


def main():
    """Time the 1M-ray aspheric trace under different Numba chunk sizes."""
    lens = common.LENSES["aspheric_double_gauss"]()
    table = common.surface_table(lens, lens.primary_wavelength)
    columns = common.ray_columns(common.launch_rays(lens, 1129))
    print(f"{columns[0].size:,} rays, threading layer: ", end="")

    def numba_fused():
        return numba_controls.trace_sequential(
            *columns, *table.args(), table.tol, table.max_iter
        )

    numba_fused()
    print(numba.threading_layer())
    results = {
        "rays": int(columns[0].size),
        "threading_layer": numba.threading_layer(),
        "seconds": {},
    }
    seconds = results["seconds"]
    seconds["numba_fused@1t"] = timed(
        "numba 1 thread", with_numba_threads(1, numba_fused)
    )
    for chunk in (0, 4096, 256):
        numba.set_parallel_chunksize(chunk)
        label = "static (default)" if chunk == 0 else f"chunksize {chunk}"
        seconds[f"numba_fused@{THREADS}t chunk={chunk}"] = timed(
            f"numba {THREADS} threads, {label}",
            with_numba_threads(THREADS, numba_fused),
        )
    numba.set_parallel_chunksize(0)

    def rust(parallel):
        return lambda: optiland_rs.trace_sequential(
            columns, *table.args(), tol=table.tol, max_iter=table.max_iter,
            parallel=parallel,
        )  # fmt: skip

    seconds["rust_fused@1t"] = timed("rust 1 thread", rust(False))
    seconds[f"rust_fused@{THREADS}t"] = timed(f"rust {THREADS} threads", rust(True))
    assert np.allclose(rust(True)()[0], numba_fused()[0], atol=1e-9, equal_nan=True)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "numba_scheduling.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
