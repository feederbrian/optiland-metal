# Small-operations experiment: can Rust beat Optiland where calls are tiny?

The first experiment (`experiments/rust_kernels`) covered big array kernels.
This one covers the opposite end: operations on a handful of rays or none,
such as paraxial quantities, single-ray and 91-ray traces, refractive indices
and merit-function evaluations, called thousands of times by an optimizer.

Nothing under `optiland/` is modified. Prototype changes to Optiland are
applied as reversible monkeypatches (`bench/patches.py`) so they can be
measured and checked against the stock code.

> **Re-checked on upstream master** (`1e05139`): the parity tests pass and
> the conclusions hold, but upstream is now 2-3x faster on small operations,
> so the speedups over "Optiland as shipped" below are about 2.5x smaller.
> See [Upstream re-check](#upstream-re-check-optilandoptiland-master-1e05139-2026-10-14).

## Answer

**Rust can make small operations thousands of times faster, but only if it
takes over a whole operation, and Numba does nearly as well.** Small
operations in Optiland are slow because of Python orchestration, not
arithmetic. Tracing 1 ray takes 10.5 ms and tracing 331 rays 12.5 ms, and
one merit evaluation makes 256,000 Python function calls to do ~5,700
ray-surface intersections.

* **Rust as a drop-in for the arithmetic: 0.9-1.15x.** Doing the surface
  math in Rust inside Optiland leaves ~90% of the time untouched.
* **Rust running the whole operation: 200-10,800x.** A flattened "compiled
  lens" evaluates `f2` in 105 ns (Optiland: 1.1 ms), one ray in 0.93 µs
  (10.1 ms), and the tutorial's full merit in 461 µs (104 ms). The whole
  tutorial optimization's work drops from 805 s to 3.7 s.
* **Numba with the same design is equivalent** for anything above ~10 µs
  (merit 497 vs 461 µs; with parallel gradients Numba was slightly faster).
  Rust's real edge is per-call overhead: 0.1 vs 0.7 µs for `f2`, 1.9 vs
  5.4 µs to pass 8 arrays. That only matters for millions of sub-microsecond
  calls from Python.
* **Most of the cost is Optiland's object model, not Python.** The same
  flattened algorithm in plain Python is already 8x faster than Optiland on
  a merit evaluation and 460x on a single ray.

| Tutorial optimization, identical work | Wall time | vs shipped | Results identical to shipped? |
|---|---:|---:|---|
| Optiland as shipped | 805 s | 1x | (reference) |
| Optiland + Python fixes (prototype) | 201 s | 4.0x | bit-identical |
| Optiland, gradients over 4 processes | 254 s | 3.2x | bit-identical |
| Optiland + fixes + 4 processes | 70 s | 11.5x | bit-identical |
| Numba compiled lens | 4.0 s | 201x | to 1e-9 |
| Rust compiled lens | 3.7 s | 216x | to 1e-9 |
| Numba compiled lens, parallel gradient batches | 1.2 s | 693x | to 1e-9 |
| Rust compiled lens, parallel gradient batches | 1.3 s | 602x | to 1e-9 |

The compiled rows replay the 7,432 designs the stock optimizer evaluated, so
every row does the same work. The "Python fixes" are pure-Python prototypes,
described under Recommendations.

### Where a small operation's time goes

Self time of one tutorial merit evaluation (`bench/characterize.py`):
NumPy calls on 1- to 91-element arrays 23%, rebuilding the paraxial path
and paraxial traces 19%, `be.*` backend dispatch 19%, refractive-index
cache keys 13%, other Python 13%, the actual ray physics 8%. Specifically:

* `trace_generic` rebuilds the paraxial path **7 times** per call during
  ray aiming (`build_paraxial_path` via `EPL` once, and twice each via
  `entrance_pupil_axial_position`, `SurfaceGroup.positions` and
  `_entry_frame`), although
  `SurfaceGroup.build_paraxial_path` itself says high-level operations
  should "build it once and pass it through their call chain".
* `BaseMaterial._create_cache_key` turns the whole wavelength array (up to
  1,024 elements, so 91 floats for a 91-ray trace) into a tuple on every
  lookup, plus a recursive state fingerprint, ~53 times per single-ray
  trace, even though every ray in a trace shares one wavelength.
* every `be.sqrt(x)` goes through a module `__getattr__`, an instance
  `getattr` and a passthrough wrapper: 1.2 µs against 0.49 µs for
  `np.sqrt` on the same 1-element array.

### Trajectory sensitivity (read before using any fast path)

The tutorial optimization is chaotic. Implementations that agree with
Optiland to 1e-12 on every merit still end in different local minima when
left to run freely: final merit **0.81** for Numba and Rust versus
**0.084** for stock Optiland, from the same start with the same settings.
Merely changing the order Rust sums four residuals moved its run from 0.83
to 0.81. The stock run's own convergence profile shows why: 92% of its
evaluations happen before its best merit is even within 2x of where it
ends, and the good basin is found near the end. So:

* Only the Python fixes and the process-parallel gradient reproduce stock
  results **bit for bit** (same 7,432 evaluations, 626 iterations, final
  merit 0.0842585). Use those if identical results matter.
* With a compiled lens, spend the speed on robustness instead.
  `bench/multistart.py` ran 32 perturbed starts in **16.5 s** (1/49 of
  one stock run). The best start reproduced the stock design (merit
  0.08425851, re-scored by Optiland); the median start ended at 1.07. A
  single run's result depends partly on luck; multi-start removes that
  dependence.

## Recommendations, best value first

1. **Fix the Python orchestration in Optiland (4x, bit-identical, no new
   dependencies).** The prototypes in `bench/patches.py` show the gain;
   the upstream versions would be:
   * build the `ParaxialPath` once per trace/aim and pass `path=` down
     (most first-order methods already accept it). Prototyped here as a
     per-operation memo scope: 3.5x on a merit evaluation by itself
     (100.9 to 28.5 ms);
   * key the material index cache on the scalar wavelength when the array
     is uniform, instead of tupling the whole array;
   * bind backend functions as module attributes in `set_backend`, so
     `be.sqrt` skips `__getattr__` (1.19x alone; 3.8-4.4x together with
     the memo scope across runs).
   Every Optiland user benefits, including analyses and GUI interaction.
2. **Parallelize finite-difference gradients across processes (3.2x, or
   11.5x with 1).** The N+1 evaluations per gradient are independent, so an
   optimizer option like `workers=4` can farm them out to processes that
   each hold their own lens. It's bit-identical, and needs no compiled
   code.
3. **For optimization-heavy work, add a compiled "frozen lens" evaluator
   (200x, 600-700x with parallel gradients).** Snapshot the lens once per
   optimization, update only the variables, and evaluate merit functions
   in one compiled call. **Use Numba**: it is already a dependency and was
   within 10% of Rust here. Budget for the costs: it is a second
   implementation of the physics for a supported subset (surface types,
   operands, field and aperture types), it needs parity tests like
   `tests/test_small_ops_parity.py`, and its optimizations follow different paths
   than stock (see above), so pair it with multi-start or a global
   optimizer.
4. **Reach for Rust only for what Numba can't do here:** sub-microsecond
   call latency in hot Python loops, a compiled core shared outside Python
   (e.g. the GUI or other languages), or SIMD transcendentals
   (experiment 1). Otherwise it adds a toolchain and platform wheels for no
   measured gain on this workload.
5. **Pin BLAS threads when running parallel kernels next to scipy.**
   OpenBLAS spins up one thread per core inside L-BFGS-B. In this
   container that made the parallel-gradient optimizations 2.7x (Rust) and
   5.9x (Numba) slower until `OPENBLAS_NUM_THREADS=1` (section G of
   `results/small_ops.md`). Serial timings are unaffected.

### What did not help

* **Rust in the leaf** (surface arithmetic only): 0.9-1.15x.
* **`LeastSquares` (trust-region-reflective)** on this problem: 534 s and a
  final merit of 8.1 on stock Optiland; the compiled lens reached the same
  ~8.1 basin in 3 s. Faster per evaluation, much worse result.
  (Levenberg-Marquardt is unavailable: 4 residuals < 7 variables.)
* **Loosening the tolerance:** the stock run finds its final basin only
  in the last ~8% of evaluations, so stopping early loses the design.

### Caveats

* One workload (the docs' Cooke-triplet case study: 7 surfaces, 91-ray
  traces) on one 4-vCPU VM. Larger ray counts shift the balance toward
  experiment 1's regime, where per-ray work dominates.
* The Python fixes are reversible prototypes, validated here for
  correctness (bit-identical merits, rays and optimizer trajectory), but
  not yet run against Optiland's test suite as real code changes. The memo
  scope assumes nothing mutates the lens during one merit evaluation,
  which holds for Optiland's operands, and it opens after the optimizer
  has applied its variables.
* `LeastSquares` evaluates operands one at a time, so under the fixes it
  only gets per-trace scopes; its "+ fixes" row understates them.
* The stock optimization was timed once (`characterize.py`, 805 s); the
  fixed Optiland runs reproduced its trajectory exactly in 200-215 s
  across three runs.
* Full tables: `results/small_ops.md`. Raw data: `results/*.json`.

## Upstream re-check (optiland/optiland master `1e05139`, 2026-10-14)

The results further down were measured on fork commit `8bd86cf`. Upstream
has since landed three changes aimed at the same overhead: #938 (build the
paraxial path once per ray generation, found independently in upstream
issue #934), #939 (cheaper path builds) and #951 (cheaper material cache
keys). Re-run on upstream (`results/upstream_*.json`):

* **All 21 parity tests still pass.** The stock tutorial optimization is
  bit-identical across versions (626 iterations, 7,432 evaluations, final
  merit 0.08425849520125625), so the recorded trajectory and the replays
  still apply.
* **Upstream is 2-3x faster on small operations:**

  | Per call | Fork `8bd86cf` | Upstream `1e05139` |
  |---|---:|---:|
  | `paraxial.f2()` | 1.06 ms | 0.54 ms |
  | `trace_generic`, 1 ray | 10.4 ms | 3.4 ms |
  | merit evaluation | 101 ms | 41 ms |
  | tutorial optimization | 805 s | 347 s |

* **The remaining overhead is now mostly materials and dispatch.** In a
  merit evaluation (119k Python calls, down from 256k): material index and
  cache keys 27% (keys still tuple the whole wavelength array), backend
  dispatch 16%, NumPy on tiny arrays 16%, ray physics 15%, paraxial path 8%.
* **The prototypes still help, by less:**

  | Merit evaluation on upstream | Time | vs upstream |
  |---|---:|---:|
  | upstream as shipped | 41.4 ms | 1x |
  | + per-operation memo scope | 24.2 ms | 1.7x |
  | + direct backend binding | 20.6 ms | 2.0x |
  | + Rust leaf as well | 16.4 ms | 2.5x |
  | Numba compiled lens | 0.52 ms | 80x |
  | Rust compiled lens | 0.46 ms | 91x |

  The tutorial optimization's work takes 3.7 s on the Rust lens and 1.3 s
  with parallel gradients, against 347 s on upstream (94x and ~260x).
* **The conclusions stand:** Rust and Numba are within ~10%, a Rust leaf
  is a small gain (1.17x on upstream), and the remaining Python fixes are
  now the material cache key (uniform-wavelength fast path) and direct
  backend binding.

## Method

### Workload

The optimization case study from Optiland's own docs
(`docs/examples/Tutorial_3d_Optimization_Case_Study_Cooke_Triplet.ipynb`):

* the tutorial's starting Cooke triplet (7 surfaces, 3 glasses, EPD 10 mm,
  angle fields 0/14/20°, 3 wavelengths)
* merit: `f2` target 50 mm, plus `rms_spot_size` at Hy = 0, 0.7, 1 over all
  three wavelengths with 5-ring hexapolar sampling (91 rays per trace,
  9 traces per merit evaluation)
* 7 variables (six radii, one thickness), Optiland's default scalers
* `OptimizerGeneric.optimize(tol=1e-9)`: scipy L-BFGS-B with
  forward-difference gradients

`bench/characterize.py` measures the stock cost of each small operation and
profiles where it goes.

### Three granularities for Rust

1. **The call boundary** (`noop`, `add3`, `echo_arrays8`): what one
   Python→Rust, Python→Numba or Python→NumPy call costs before doing work.
2. **A drop-in leaf** (`trace_surface`, `patches.rust_surface_leaf`): one
   surface of `Surface._trace_real` (intersection, propagation, absorption,
   OPD, refraction) in Rust. Everything else in Optiland stays as it is.
3. **A compiled lens** (`CompiledLens`, `src/lens.rs`): the lens and merit
   function are flattened once into plain arrays (`bench/snapshot.py`). Each
   call then runs the whole operation in compiled code: paraxial `f2` and
   entrance pupil, Optiland's paraxial ray aiming, the real trace, the
   multi-wavelength RMS spot, and the weighted merit sum.

As in experiment 1, every Rust version has controls running the *same
algorithm*, so the language can be separated from the restructuring:
plain Python (`lean.PythonLens`), lean NumPy (`lean.NumpyLens`) and Numba
(`numba_lens.py`).

### Correctness

* All compiled-lens implementations match Optiland's `f2`, EPL, single-ray
  traces (every wavelength), RMS spots, residuals and merit to 1e-10 or
  better (`tests/test_small_ops_parity.py`, 21 tests).
* The end-to-end comparison replays the stock optimizer's 7,432 evaluation
  points through every implementation, and every replayed merit matches
  Optiland's to 1e-9.
* Every Optiland patch is checked to leave results **bit-identical**: same
  merits, same rays, and the same optimizer trajectory (7,432 evaluations,
  626 iterations, identical final merit).

### Scope of the compiled lens

Centered, untilted planes and conics; refraction only; EPD aperture; angle
fields with an object at infinity; no vignetting; `f2` and
`rms_spot_size(wavelength="all")` operands; radius and thickness variables
with linear scalers. `bench/snapshot.py:capture` refuses anything else.
Glass indices are captured at snapshot time, so glass variables would
need a re-snapshot.

## Reproduce

From the repository root, with the experiment-1 setup (`uv sync`,
`uv pip install maturin`):

```bash
cd experiments/rust_small_ops
VIRTUAL_ENV=../../.venv maturin develop --release --uv
cd ../..

.venv/bin/python -m pytest experiments/rust_small_ops/tests -v
.venv/bin/python experiments/rust_small_ops/bench/characterize.py    # ~15 min (stock optimization)
.venv/bin/python experiments/rust_small_ops/bench/run_small_ops.py   # ~15 min
.venv/bin/python experiments/rust_small_ops/bench/explore_options.py # ~20 min
.venv/bin/python experiments/rust_small_ops/bench/multistart.py      # ~20 s
.venv/bin/python experiments/rust_small_ops/bench/report.py
```

Both `characterize.py` and `run_small_ops.py` take `--out NAME` to write
to another file in `results/` (the upstream re-check used `upstream_*.json`).

`run_small_ops.py --skip-slow` leaves out the multi-minute plain-Python and
lean-NumPy replays.

## Layout

```
src/lens.rs                 flattened lens: paraxials, aiming, trace, RMS, merit
src/lib.rs                  PyO3 bindings: boundary probes, trace_surface, CompiledLens
bench/workload.py           tutorial lens + merit function, timing helpers
bench/snapshot.py           Optic + problem -> plain data (refuses out-of-scope input)
bench/lean.py               plain-Python and lean-NumPy compiled-lens controls
bench/numba_lens.py         Numba compiled-lens control
bench/patches.py            reversible Optiland prototypes (Rust leaf, memoization, backend)
bench/characterize.py       stock latencies, ray-count sweep, profile, stock optimization
bench/run_small_ops.py      sections A-E and G (BLAS)
bench/explore_options.py    optimizer choice and process-parallel gradients (section F)
bench/multistart.py         multi-start search on the compiled lens (section H)
bench/report.py             JSON -> results/small_ops.md
tests/test_small_ops_parity.py
```
