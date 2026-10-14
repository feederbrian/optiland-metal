# UW2 lens workload: exact speedups where the optimizer spends its time

The previous experiments measured Optiland on a textbook triplet. This one
targets the real workload: the UW2 lens and the merit function its
optimizer evaluates. The lens and the pipeline's job files are not in this
repository. The harness loads a `uw1-optic/1` export with `--optic PATH`
or the `UW2_OPTIC` environment variable (`bench/uw1_optic.py`), and checks
every load against the export's own vertex positions and refractive
indices.

Base: upstream master `1e05139` (branch `upstream-master`). All changes
under `optiland/` are on this branch, `experiment/uw2-workload`.

## Answer

**One finite-difference Jacobian column of the merit function is 2.5x
faster (17.3 s to 6.8 s) and every result is bit-identical to stock
Optiland.** Per column, 648 operand values were compared, NaNs included.
The robust ray aimer took 94% of an operand's time after each variable
step. Five changes inside Optiland remove most of that work without
changing a single bit:

| One step of the optimizer, this VM | Stock | Optimized | Speedup |
|---|---:|---:|---:|
| FD Jacobian column, full proxy merit (mean of 8) | 17.26 s | 6.77 s | 2.55x |
| Merit evaluation after a step ahead of the stop | 17.58 s | 6.79 s | 2.59x |
| Spot, 5 wavelengths, after a step | 964 ms | 424 ms | 2.27x |
| Wavefront RMS, one field and wavelength, after a step | 266 ms | 172 ms | 1.55x |
| Chief ray, after a step | 173 ms | 136 ms | 1.27x |
| Wavefront RMS, lens unchanged | 37.5 ms | 29.6 ms | 1.27x |
| Equivalence harness (`bench/equivalence.py`) | 47.4 s | 23.1 s | 2.05x |

Single operands gain less than a full merit because the largest saving is
batching across the many fields and wavelengths that one merit evaluation
requests. The pipeline's V2 budget record names time as the binding
limit: it chose 128 Gauss-Newton steps against a nominal 648 (8 per
variable), at 9.3 s per evaluation. So a faster evaluation turns directly
into more steps per stage, or a shorter stage.

## The workload

From the job files:

* **Optimizer:** block Gauss-Newton/trust-region, 81 variables, 92
  operands, finite-difference Jacobian (relative step 1e-5) farmed out to
  6 worker processes.
* **Operands:** RMS wavefront at 4 fields x 5 wavelengths (20); RMS spot
  (4); 32 tolerance-sensitivity terms (16 perturbations x 2 fields); a
  second F/2.8 configuration (5 spot, 2 MTF); field MTF; chief-ray angle,
  framing and focal-length terms; corner-ray survival; and geometric
  constraints (track, rear focus, diameters, edge and air gaps, center
  thickness).
* **Lens:** robust ray aiming (`max_iter` 60, `tol` 1e-9, with the aimer
  cache on), even aspheres before and after the stop, and an image-space
  F-number aperture, so the entrance pupil depends on every surface.

`bench/uw2_operands.py` reproduces each traced operand kind with the
public Optiland call it most likely rests on, at the job's sampling. The
pipeline's own operand code was not available, so this is a proxy for
where the time goes, not for the residual values.

## Where the time went (stock)

After any variable step ahead of the stop, every field's cached pupil map
is stale. The first request for each (field, wavelength) then pays for:

1. **A real marginal-ray solve for the stop radius, on every aiming call:**
   about 32 ms, repeated by every one of the ~136 aiming calls in a merit
   evaluation, although its value depends only on the system.
2. **A calibration, one field at a time:** a Newton chief-ray solve and
   four edge-probe solves. That is 57 calibrations per merit evaluation,
   36% of its time.
3. **A Newton polish of the requested pupil points, one call at a time:**
   136 per evaluation, about 35%.

Under all three, every trace is Python orchestration on 1 to 127 rays.
Refractive-index lookups alone were 26-36% of the self time: each re-hashed
the material's whole state and turned the wavelength array into a tuple
key, several times per surface.

## The changes

All five are exact: they skip or batch work whose result is fully
determined, and each is checked bit for bit.

1. **Stop radius once per system state** (`robust.py: _stop_radius`,
   `_system_state_key`). Its value is a deterministic function of the
   system, so it is reused while an exact serialization of the state is
   unchanged: surfaces, fields, wavelengths and aperture, pickled as
   `CachedRayAimer` does, plus every material's optical-state token. Any
   change, down to the last bit or an in-place coefficient write, solves it
   again.
2. **Frozen-materials scope** (`materials/base.py`). Every trace and
   aiming call runs in a scope that promises not to change materials, as
   upstream's paraxial `shared_path_scope` already does for geometry.
   Inside the scope each material fingerprints its state once, and a lookup
   whose wavelength array has the same bytes, shape and dtype as that
   property's previous lookup returns that result directly. Values still
   come from the material's own cache.
3. **Batched calibration ahead of request** (`robust.py:
   _speculative_map`, `_calibrate_batch`). Each ray's Newton solve is
   independent of the rest of its batch, so the chief solves of many fields
   run as one solve, then all their edge probes as another. Each field gets
   exactly the map its own calibration would. A batch costs about one
   field's calibration: 24 chief solves took 23 ms batched against 425 ms
   one at a time. The maps are only committed when their field is
   requested, so other fields see exactly the warm starts they see in
   stock, and maps for fields never requested are dropped.
4. **Replayed batched polish** (`robust.py: _take_polish`,
   `_speculative_polish`, `_polish_batch`; `iterative.py: ray_info`). An
   optimizer asks for the same rays after every step. When a new system
   state begins with the same request as an earlier one, that state's
   remembered requests are polished in batches, grouped by finite-difference
   step scale and wavelength. A result is served only to a request with the
   same field, pupil points and wavelengths whose map is the same object.
   Per-request solve reports are rebuilt from per-ray bookkeeping, so
   `last_report` matches stock as well.
5. **Even-asphere normal** (`even_asphere.py`). Each `r2**i` is computed
   once and shared by both slopes, with the same operation order.

Safeguards:

* A batch whose Newton solve needed the paraxial-Jacobian fallback is
  discarded, because that fallback uses the batch's mean wavelength.
  Those fields are solved one at a time, as in stock.
* Work done ahead never raises. A field whose seed fails is left to its
  own calibration, which raises exactly as stock does.
* Batching runs only on the NumPy float64 backend with an exact state key.
  `set_aiming("robust", ..., solve_ahead=False)` turns it off; results are
  identical either way.

## Correctness

* **This lens, `bench/equivalence.py`:** a base point, one-sided steps on
  radii, thicknesses and an aspheric coefficient before and after the
  stop, a return to base, an accepted multi-variable step, and two full
  proxy merits. 111 quantities and 9,333 values (operands plus raw
  image-plane rays) are identical to stock. The stock reference reproduces
  itself bit for bit between runs.
* **This lens, `bench/fd_loop.py`:** 8 Jacobian columns of the full proxy
  merit; 648 values identical.
* **New Optiland tests:**
  * `tests/test_ray_aiming_ahead.py` replays steps, restores and an
    aperture change on five sample lenses (Cooke triplet, finite-conjugate
    triplet, 100 and 120 degree wide-angles, aspheric singlet), with and
    without the aimer cache. Rays and aimer reports must be identical with
    `solve_ahead` on and off, and the batched paths must actually run. It
    also checks that the stop-radius memo equals a fresh solve after every
    change, including an in-place coefficient write.
  * `tests/test_material_frozen_scope.py` checks values against unscoped
    lookups for three material kinds and six query types, in-place
    wavelength writes, material changes between scopes, and nesting.
* **Full suite:** 6,819 passed (the 18 new tests among them) and 509
  skipped. Two tests were excluded or failed, and both fail identically on
  stock in this VM:
  * `test_large_array_cache_key_avoids_python_scalar_overhead`, a timing
    ratio (about 160 here, limit 40);
  * `test_compatibility_matrix`, which fails only under pytest-xdist
    (`can't serialize <class 'function'>`).

  GUI tests were not run because PySide6 is not installed.

**Stock is history-dependent, and the optimized code keeps that exactly.**
The robust aimer warm-starts from the previous state's maps, so an
operand's last bits depend on which state the aimer saw before. On this
lens that is about 1e-11 waves, or 2.5e-8 relative noise in a
finite-difference derivative. The equivalence runs replay identical call
histories. A change that altered the history would not be bit-identical,
and none of the five does.

## What did not make the cut

* **Binding backend functions into the module namespace** (an estimated
  10%, from the small-operations experiment):
  it breaks tests that monkeypatch a backend instance or
  `be._current_backend`, and user code that does the same.
* **`record=False` for the aimer's internal traces** (3-7%): ray results
  are identical, but the per-surface records left between aiming and the
  main trace would change.
* **Faster but inexact aiming:** seeding edge probes from the stale map,
  skipping recalibration after small steps, or the compiled lens from the
  small-operations experiment. These change results at the aimer's
  tolerance level. If that is ever acceptable, stock's own history noise
  of about 1e-11 waves is the bar to compare against.

## Next steps, best value first

1. **Run the pipeline on this branch.** Nothing in the pipeline has to
   change. The pipeline's runtime patches were written for 0.6.2 and need
   checking against upstream before use. Upstream's Newton-Raphson already
   freezes converged rays and stops on NaN residuals, and #938 already
   shares the paraxial path per ray generation, which may make some of
   them unnecessary.
2. **Trace several fields per call where the pipeline can.**
   `Optic.trace` accepts arrays of field coordinates. The remaining third
   of the time is 157 separate full traces per merit evaluation, each
   Python orchestration on at most a few hundred rays.
3. **Upstream these changes.** They are self-contained (`robust.py`,
   `iterative.py`, `pupil_map.py`, `materials/base.py`, two scope entries,
   `even_asphere.py`) and come with their own tests.

## Reproducing

```bash
# stock reference: any checkout of upstream-master
git worktree add --detach ../optiland-ref upstream-master
export UW2_OPTIC=/path/to/uw2A_X14gs_dual28n3_s1_V2.optic.json OPENBLAS_NUM_THREADS=1
B=experiments/uw2_workload/bench
PYTHONPATH=../optiland-ref .venv/bin/python $B/fd_loop.py --out stock_fd.npz
.venv/bin/python $B/fd_loop.py --out new_fd.npz
.venv/bin/python $B/equivalence.py --compare stock_fd.npz new_fd.npz
PYTHONPATH=../optiland-ref .venv/bin/python $B/equivalence.py --out stock.npz
.venv/bin/python $B/equivalence.py --out new.npz
.venv/bin/python $B/equivalence.py --compare stock.npz new.npz
.venv/bin/python $B/bench_operands.py --out operands.json
.venv/bin/python $B/profile_operand.py --operand spot --mode perturbed
```

Run the scripts from outside the repository root, or by path as above, so
that `PYTHONPATH` decides which Optiland is imported. Each script prints
the path it loaded. The `.npz` outputs hold lens data and are not
committed. `results/` holds the timings only.

Caveats: one lens and one 4-vCPU VM (the pipeline runs on an M1 Max); a
proxy for the pipeline's operands; timings on this VM vary by about 5%
between runs.
