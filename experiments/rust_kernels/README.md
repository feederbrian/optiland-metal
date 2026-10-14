# Rust kernels experiment

Is Rust faster than Optiland's current Python stack (NumPy, SciPy, and the
Numba kernels it already ships) on its heaviest operations?

This directory is self-contained. It is **not** part of the `optiland`
package build: nothing under `optiland/` imports it, and the main
`pyproject.toml` / `uv.lock` are unchanged.

> **Upstream note (2026-10-14).** Results below were measured on fork
> commit `8bd86cf`. On `optiland/optiland` master (`1e05139`), upstream #864
> moved the Huygens obliquity normal to the reference-sphere center; the
> Rust and Numba kernels here follow it (`reference_center` argument), all
> 25 parity tests pass against upstream, and Huygens timings are unchanged.

## Answer

**For the same algorithm, Rust is not faster than Numba.** The literal
Huygens port and the fused tracers ran within about 5% of their Numba twins
on every case above 10 ms, single- and multi-threaded. The scalar "hoisted"
Huygens variant was actually 13-41% *slower* in Rust, because Numba compiles
it with `fastmath` (reassociation lets LLVM vectorize the reduction) while
Rust keeps strict IEEE ordering. On tiny batches (~1k rays, 4 threads) Rust's
~0.3 ms per-call overhead also loses to Numba.

Large speedups over today's Optiland are real, but they come from
restructuring the computation, which Numba (already a dependency) can do too.
Rust pulled clearly ahead in one place: SIMD-vectorized `sin`/`cos` in the
Huygens kernel, 1.3-1.7x over the best Numba version.

| Kernel (largest case) | Shipped | Best Numba | Best Rust | Rust vs best Numba |
|---|---:|---:|---:|---:|
| Huygens PSF, 128x128 image, 4 threads | 2,838 ms | 1,143 ms (2.5x) | 674 ms (4.2x) | 1.7x |
| Huygens PSF, same, 1 thread | 10,892 ms | 3,722 ms (2.9x) | 2,644 ms (4.1x) | 1.4x |
| Real-ray trace, 1M rays, aspheric Double Gauss, 4 threads | 4,377 ms | 340 ms (12.9x) | 332 ms (13.2x) | 1.0x |
| Real-ray trace, same, 1 thread | 4,377 ms | 1,304 ms (3.4x) | 1,302 ms (3.4x) | 1.0x |
| 2-D FFT 1024x1024, 4 threads | 23.1 ms | 7.75 ms (SciPy, 3.0x) | 7.70 ms (3.0x) | 1.0x |

What each result means:

1. **Huygens-Fresnel PSF** (the heaviest single operation: 88% of a
   `HuygensPSF` call). The literal Rust port matches Optiland's Numba kernel
   (2,964 vs 2,838 ms). Hoisting the pupil phase out of the inner loop gives
   2.3-3.5x in Numba with a ~10-line change. A vectorizable `sin_cos` then
   gets Rust to 4.2-5.5x. The same branch-free `sin_cos` written in Numba
   only reached 2.5-4.0x: Numba cannot express FMA (no `math.fma`), and its
   lane loop evidently did not vectorize (only 3-13% faster than the scalar
   hoisted version; inferred from timings, not from the assembly). Numba's
   usual fix, Intel SVML, is not available on Optiland's install path: the PyPI
   `llvmlite` wheel is not built against SVML-patched LLVM, so installing
   `icc_rt` does nothing (`numba -s` reports "SVML Operational: False").
2. **Real-ray surface loop.** Fusing localize, intersect, propagate, refract
   and globalize into one ray-major compiled loop beats the NumPy pipeline
   by 1.5-3.4x on one thread at 10k+ rays and 10-15x on small batches, where
   NumPy per-call overhead dominates. Threads raise that to 4.5-13x. Rust and
   Numba are tied throughout (1M rays: 761 vs 784 ms spherical, 1,302 vs
   1,304 ms aspheric). Run 1 suggested Rust scaled better on 4 threads, but
   it did not reproduce, and `bench/numba_scheduling.py` shows dynamic
   chunking changes Numba by only ~7%.
3. **FFT.** rustfft roughly matches pocketfft on one thread and matches
   `scipy.fft.fft2(workers=4)` on 4 at 512-1024 (slower at 256 and 2048).
   The free win is SciPy's `workers=` argument, but FFT is only 11-12% of an
   FFT PSF/MTF.
4. **End to end**, with the Rust kernel swapped into an unmodified Optiland
   call: `ScalarHuygensPSF` 3,060 to 886 ms (3.5x); `lens.trace` of 205k
   rays 2.9x (spherical) and 4.0x (aspheric).

### Where the time actually goes

`bench/profile_breakdown.py`, Double Gauss, share of profiled time
(cProfile adds overhead, so Python shares are somewhat overstated):

| Analysis | Wall | Real-ray surface loop | Kernel (Huygens / FFT) | Python orchestration |
|---|---:|---:|---:|---:|
| `HuygensPSF`, 128 rays, 128 px | 3,298 ms | 3% | 88% | 9% |
| `lens.trace`, 205k rays | 444 ms | 75% | - | 25% |
| `FFTMTF`, 256 rays, 1024 grid | 928 ms | 28% | 12% | 59% |
| `FFTPSF`, 256 rays, 1024 grid | 192 ms | 32% | 11% | 57% |
| `SpotDiagram`, 3 fields x 3 wavelengths | 270 ms | 28% | - | 72% |
| `GeometricMTF` | 174 ms | 23% | - | 77% |
| `OPD` map | 48 ms | 23% | - | 77% |

For spot diagrams, OPD and geometric MTF, about three quarters of the time
is Python bookkeeping (paraxial path construction, material cache keys,
backend `__getattr__` dispatch), so even an infinitely fast kernel tops out
around 1.3-1.4x. That overhead, not arithmetic, is the next bottleneck there.

### Recommendations, cheapest first

1. **Hoist the pupil phase in `NumbaSummation`.** About 10 lines, no new
   dependency: 2.3-3.5x on Huygens PSF/MTF. (`numba_hoisted` in
   `bench/numba_controls.py` is a drop-in, verified against the shipped
   kernel.)
2. **Add a fused Numba trace path** for the common case on the NumPy
   backend (plane / conic / even asphere, no aperture), next to the existing
   `_numpy_conic_distance` fast path. That gives 1.5-3.4x on one thread and
   4.5-13x with threads at 10k+ rays. The general NumPy path stays
   for Torch/autograd and other surface types.
3. **Use `scipy.fft` with `workers=`** in the FFT PSF/MTF: a one-line change,
   1.8-3.6x on the FFT portion.
4. **Cut Python orchestration** in the analysis layer. That is the biggest
   lever for spot / OPD / geometric MTF, and no compiled kernel touches it.
5. **Adopt Rust only for the last ~1.5x on transcendental-heavy kernels**
   like Huygens, and only if the project will ship per-platform wheels
   (maturin + cibuildwheel) and maintain a second language. Like Numba,
   Rust kernels serve only the NumPy backend; the Torch/autograd path stays
   Python either way.

### Caveats

* One 4-vCPU cloud VM (Xeon @ 2.8 GHz, AVX-512). Across two full runs the
  median cell changed by 3% (5th-95th percentile 0.69-1.12x); sub-millisecond
  cells are the noisiest, and the conclusions above rest on 10 ms+ cells.
* Both compilers target the host CPU (`target-cpu=native` for Rust).
* The trace port covers centered refractive systems only (see Scope below).
* Full tables, including every size: `results/benchmarks.md`. Raw data:
  `results/*.json`.

## What was measured

Profiling the stock analyses on the Double Gauss sample
(`bench/profile_breakdown.py`) picked out three compute kernels:

| Kernel | Where Optiland runs it | Current implementation |
|---|---|---|
| Huygens-Fresnel summation | `HuygensPSF`, `HuygensMTF` | Numba, `parallel=True, fastmath=True` |
| Real-ray surface loop | every real trace (`SurfaceGroup.trace`) | NumPy, one whole-array op at a time |
| 2-D FFT | `FFTPSF`, `FFTMTF` | `numpy.fft` (pocketfft, C++) |

Each was ported to Rust (`src/`), and, as a control, the same algorithm was
also written in Numba (`bench/numba_controls.py`). Without the control, "Rust
is faster than NumPy" would really mean "one fused compiled loop beats a
chain of NumPy temporaries", which Numba can do too. Rows are labelled:

* **shipped**: Optiland's code as it is today; the baseline.
* **literal port**: the same arithmetic in Rust, statement for statement.
* **hoisted**: `exp(-i k opd_j)` depends only on the pupil point, so it is
  computed once per pupil point instead of once per (image, pupil) pair.
  This halves the transcendental work and is available to any language.
* **fast sincos / SIMD**: a branch-free `sin_cos` (Cody-Waite reduction plus
  the fdlibm polynomials, `src/fastmath.rs`) with 8-lane accumulators, so
  LLVM can emit AVX-512. Standard `sin`/`cos` are scalar libm calls in both
  Rust and Numba (without Intel SVML), which blocks vectorization.
* **fused**: the trace carries each ray through every surface in one pass
  (ray-major) instead of one NumPy operation per step over all rays.

### Correctness

Every implementation is checked against Optiland's own output before it is
timed, and `tests/test_parity.py` (25 tests) pins this down:

* Spherical systems: the Rust trace is **bit-identical** to Optiland in
  position, direction and OPD (intensity differs in the last ulp of `exp`).
* Aspheres: agreement to ~1e-13 mm, well inside the 1e-10 mm Newton
  tolerance (the polynomial uses Horner's rule instead of powers).
* Huygens: within 1e-9 of the PSF peak for every variant (the shipped
  kernel uses `fastmath`, so its own summation order is not fixed).
* FFT: matches `numpy.fft.fft2` to round-off.

### Scope of the trace port

Centered, untilted systems of planes, standard conics and even aspheres;
refraction with absorption; no physical apertures, mirrors, coatings or
polarization. `bench/common.py:surface_table` refuses anything else. The
benchmark lenses are the Double Gauss and aspheric singlet samples, plus an
"aspheric Double Gauss" (every curved surface given small 4th-8th order
terms) so Newton-Raphson has real work on every surface.

One deviation: Optiland's Newton convergence floor uses the largest `|t|`
in the batch; the fused loop uses each ray's own `|t|`. In float64 the
1e-10 tolerance dominates both unless distances exceed ~50 m.

## Reproduce

From the repository root:

```bash
uv sync                                  # Optiland's normal dev environment
uv pip install maturin
cd experiments/rust_kernels
VIRTUAL_ENV=../../.venv maturin develop --release --uv
cd ../..

.venv/bin/python -m pytest experiments/rust_kernels/tests -v
.venv/bin/python experiments/rust_kernels/bench/run_benchmarks.py      # ~5 min on 4 cores
.venv/bin/python experiments/rust_kernels/bench/report.py
.venv/bin/python experiments/rust_kernels/bench/profile_breakdown.py
```

`--quick` gives a one-minute smoke run. For the SVML comparison:
`uv pip install icc_rt`, then
`NUMBA_CACHE_DIR=$(mktemp -d) run_benchmarks.py --only huygens --tag svml`.

Rust unit tests (fast `sin_cos` against libm over ±2e6 rad):

```bash
cd experiments/rust_kernels
PYO3_PYTHON=../../.venv/bin/python \
LD_LIBRARY_PATH=$(../../.venv/bin/python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") \
cargo test --release
```

`.cargo/config.toml` builds with `target-cpu=native`, matching Numba, which
always compiles for the host CPU. Drop it for a portable wheel.

## Layout

```
src/lib.rs          PyO3 bindings (GIL released during compute)
src/huygens.rs      literal, hoisted and SIMD Huygens-Fresnel summation
src/fastmath.rs     branch-free, vectorizable sin_cos
src/trace.rs        fused ray-major sequential tracer
src/fft.rs          rustfft-based fft2 with parallel passes
bench/common.py     lenses, surface tables, Optiland reference calls, timing
bench/numba_controls.py  the same algorithms in Numba
bench/run_benchmarks.py  timing harness -> results/benchmarks*.json
bench/report.py     JSON -> Markdown tables
bench/profile_breakdown.py  kernel vs Python-overhead split per analysis
tests/test_parity.py
results/            raw JSON and rendered tables from the run quoted above
```
