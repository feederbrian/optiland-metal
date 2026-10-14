Machine: Intel(R) Xeon(R) Processor @ 2.80GHz, 4 threads. Python 3.13.16, NumPy 2.3.5, Numba 0.65.1, rustc 1.97.0 (2d8144b78 2026-07-07). Run 2026-10-14.

### Baseline: Optiland's small operations (Cooke triplet)

| Operation | Per call |
|---|---:|
| paraxial.EPD() | 553 ns |
| paraxial.f2() | 1.1 ms |
| paraxial.EPL() | 1.1 ms |
| paraxial.XPL() | 1.0 ms |
| material.n(wl) | 23.8 µs |
| trace_generic, 1 ray | 10.3 ms |
| trace, 91 rays (5-ring hexapolar) | 10.7 ms |
| updater.update() | 1.1 µs |

| Rays traced | Per call | Per ray |
|---|---:|---:|
| 1 | 10.5 ms | 10.5 ms |
| 7 | 10.1 ms | 1.4 ms |
| 37 | 10.0 ms | 269 µs |
| 331 | 12.5 ms | 37.8 µs |
| 2,791 | 15.1 ms | 5.4 µs |
| 30,301 | 47.1 ms | 1.6 µs |
| 270,901 | 387.9 ms | 1.4 µs |

**Where `trace_generic, 1 ray` spends its time** (27,812 Python calls per call)

| Category (self time) | Share |
|---|---:|
| NumPy (Python + C) | 23% |
| paraxial path + paraxial traces | 20% |
| backend dispatch (be.*) | 18% |
| Python builtins + other | 14% |
| material index + cache keys | 13% |
| real-ray physics (Python side) | 8% |
| coordinate frames | 3% |
| ray generation + aiming | 1% |
| other Optiland Python | 0% |

**Where `problem.sum_squared()` spends its time** (256,407 Python calls per call)

| Category (self time) | Share |
|---|---:|
| NumPy (Python + C) | 23% |
| paraxial path + paraxial traces | 19% |
| backend dispatch (be.*) | 19% |
| material index + cache keys | 13% |
| Python builtins + other | 13% |
| real-ray physics (Python side) | 8% |
| coordinate frames | 3% |
| ray generation + aiming | 2% |
| other Optiland Python | 0% |
| optimization bookkeeping | 0% |

Tutorial optimization as shipped: 626 iterations, 7,432 merit evaluations, 804.9 s (108.3 ms per evaluation), final merit 0.08426.

### A. Cost of one call, before any work

| Call | Per call |
|---|---:|
| Python function, no args | 26 ns |
| Python function, 3 floats | 68 ns |
| Rust (PyO3), no args | 36 ns |
| Rust (PyO3), 3 floats | 93 ns |
| Rust (PyO3), 8 arrays of 1 in, 8 out | 1.9 µs |
| Rust (PyO3), 8 arrays of 91 in, 8 out | 2.1 µs |
| Numba, no args | 40 ns |
| Numba, 3 floats | 193 ns |
| Numba, 8 arrays of 1 in, 8 out | 5.4 µs |
| Numba, 8 arrays of 91 in, 8 out | 5.3 µs |
| NumPy a + b, 1 element | 474 ns |
| NumPy np.sqrt(a), 1 element | 487 ns |
| Optiland be.sqrt(a), 1 element | 1.2 µs |
| math.sqrt(x), Python float | 47 ns |

### B. Rust leaf: surface arithmetic in Rust, Optiland orchestration as is

| Operation | Stock | Rust leaf | Speedup |
|---|---:|---:|---:|
| trace_generic, 1 ray | 10.5 ms | 9.3 ms | 1.1x |
| trace, 91 rays | 10.6 ms | 10.1 ms | 1.1x |
| merit evaluation (optimizer._fun) | 105.5 ms | 92.0 ms | 1.1x |

### B2. Optiland prototypes, alone and combined (results bit-identical)

| Change | merit at new x | trace_generic, 1 ray |
|---|---:|---:|
| stock | 100.9 ms (1.0x) | 9.6 ms (1.0x) |
| Rust leaf | 89.1 ms (1.1x) | 10.2 ms (0.9x) |
| build once per operation (memo scope) | 28.5 ms (3.5x) | 4.6 ms (2.1x) |
| direct backend binding | 84.8 ms (1.2x) | 9.0 ms (1.1x) |
| memo scope + backend binding | 26.3 ms (3.8x) | 3.8 ms (2.6x) |
| all three | 19.9 ms (5.1x) | 3.4 ms (2.8x) |

### C. Per-operation latency (speedup over Optiland as shipped)

| Implementation | f2 | trace_generic, 1 ray | rms_spot_size, 3 wl x 91 rays | merit at new x |
|---|---:|---:|---:|---:|
| Optiland (shipped) | 1.1 ms (1.0x) | 10.1 ms (1.0x) | 32.6 ms (1.0x) | 103.6 ms (1.0x) |
| Optiland + Python fixes | 865 µs (1.3x) | 3.7 ms (2.7x) | 14.7 ms (2.2x) | 23.6 ms (4.4x) |
| Plain Python, compiled-lens algorithm | 1.8 µs (611x) | 22.0 µs (458x) | 4.5 ms (7.2x) | 13.5 ms (7.7x) |
| Lean NumPy, compiled-lens algorithm | 1.7 µs (653x) | - | 2.4 ms (14x) | 6.9 ms (15x) |
| Numba | 702 ns (1,550x) | 2.1 µs (4,867x) | 178 µs (183x) | 497 µs (209x) |
| Rust | 105 ns (10,330x) | 929 ns (10,834x) | 156 µs (209x) | 461 µs (225x) |

### D. One finite-difference gradient (8 merit evaluations)

| Implementation | Time | Speedup |
|---|---:|---:|
| Optiland (shipped), sequential | 827.8 ms | 1.0x |
| Plain Python, sequential | 117.7 ms | 7.0x |
| Numba, 1 thread | 3.9 ms | 211x |
| Numba, 4 threads | 1.1 ms | 781x |
| Rust, 1 thread | 3.7 ms | 223x |
| Rust, 4 threads | 1.2 ms | 706x |

### E. The whole tutorial optimization, same work for everyone

The 7,432 merit evaluations of the tutorial's L-BFGS-B run (626 iterations, final merit 0.0842585), recorded from Optiland and replayed through each implementation; every replayed merit matched Optiland's to 1e-9. Optiland rows are full optimizer runs on that same trajectory.

| Implementation | Wall time | Speedup |
|---|---:|---:|
| Optiland (shipped) | 804.9 s | 1.0x |
| Optiland + Python fixes | 200.6 s | 4.0x |
| Plain Python, compiled-lens algorithm | 113.3 s | 7.1x |
| Lean NumPy, compiled-lens algorithm | 58.9 s | 14x |
| Numba | 4.0 s | 201x |
| Rust | 3.7 s | 216x |
| Numba, gradient batches in parallel (4 threads) | 1.2 s | 693x |
| Rust, gradient batches in parallel (4 threads) | 1.3 s | 602x |

**How much of that run was needed.** Evaluations before the best merit so far came within a given margin of where the run finally stopped:

| Best merit so far | Evaluations | Share of run |
|---|---:|---:|
| within 100.0% of the final merit | 6,865 | 92% |
| within 10.0% of the final merit | 6,977 | 94% |
| within 1.0% of the final merit | 7,161 | 96% |
| within 0.1% of the final merit | 7,201 | 97% |

**Free-running optimizations.** Same start and settings, but each run follows its own path (see the trajectory-sensitivity note):

| Run | Wall time | Speedup | Merit evals | Final merit (Optiland) |
|---|---:|---:|---:|---:|
| Numba | 2.8 s | 293x | 4,264 | 0.81462 |
| Rust | 2.4 s | 330x | 4,264 | 0.81462 |
| Numba + parallel gradient (4 threads) | 729.1 ms | 1,104x | 4,264 | 0.81462 |
| Rust + parallel gradient (4 threads) | 897.9 ms | 896x | 4,264 | 0.81462 |

### F. Options that change how many evaluations run, or where

| Run | Wall time | Speedup | Merit evals | Final merit (Optiland) |
|---|---:|---:|---:|---:|
| Optiland LeastSquares (trf) | 534.3 s | 1.5x | 4,928 | 8.1148 |
| Optiland LeastSquares (trf) + Python fixes | 222.7 s | 3.6x | 4,928 | 8.1148 |
| Rust compiled lens, least_squares (trf) | 3.0 s | 265x | 4,998 | 8.0785 |
| Rust compiled lens, least_squares (trf), parallel Jacobian | 1.8 s | 449x | 5,612 | 8.0785 |
| Optiland L-BFGS-B, gradients over 4 processes | 254.2 s | 3.2x | 7,432 | 0.084258 |
| Optiland L-BFGS-B, 4 processes + Python fixes | 70.1 s | 11x | 7,432 | 0.084258 |

### H. Multi-start: 32 perturbed starts (sigma 0.1 in scaled units), Numba lens + parallel gradient: 16.5 s in total; best merit 0.084258513 (Optiland re-score; stock run 0.084258495), median 1.075.

### G. OpenBLAS threads vs parallel kernels (free-running optimization)

| Parallel-gradient run | OPENBLAS_NUM_THREADS=4 | OPENBLAS_NUM_THREADS=1 |
|---|---:|---:|
| Rust | 2.4 s | 898.9 ms |
| Numba | 4.8 s | 807.3 ms |
