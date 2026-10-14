Machine: Intel(R) Xeon(R) Processor @ 2.80GHz, 4 threads. Python 3.13.16, NumPy 2.3.5, SciPy 1.16.3, Numba 0.65.1 (SVML off), rustc 1.97.0 (2d8144b78 2026-07-07). Run 2026-10-14.

Cells: best wall time in ms (speedup over the baseline).

### Huygens-Fresnel PSF summation

Baseline: Optiland's shipped Numba kernel on 4 threads.

| Implementation | 64x64 image, 3,096 pupil pts | 128x128 image, 12,644 pupil pts |
|---|---:|---:|
| Optiland Numba (shipped), 4 threads | 217 (1.0x) | 2,838 (1.0x) |
| Rust, literal port, 4 threads | 192 (1.1x) | 2,964 (1.0x) |
| Numba, hoisted phase, 4 threads | 62.6 (3.5x) | 1,250 (2.3x) |
| Rust, hoisted phase, 4 threads | 88.2 (2.5x) | 1,417 (2.0x) |
| Numba, hoisted + fast sincos, 4 threads | 53.7 (4.0x) | 1,143 (2.5x) |
| Rust, hoisted + fast sincos (SIMD), 4 threads | 39.3 (5.5x) | 674 (4.2x) |
| Optiland Numba (shipped), 1 thread | 757 (0.3x) | 10,892 (0.3x) |
| Rust, literal port, 1 thread | 738 (0.3x) | 11,475 (0.2x) |
| Numba, hoisted phase, 1 thread | 243 (0.9x) | 3,849 (0.7x) |
| Numba, hoisted + fast sincos, 1 thread | 211 (1.0x) | 3,722 (0.8x) |
| Rust, hoisted + fast sincos (SIMD), 1 thread | 166 (1.3x) | 2,644 (1.1x) |

### Sequential real-ray trace (surface loop only)

Baseline: Optiland's NumPy surface loop (`SurfaceGroup.trace`, `record=False`).

**double_gauss** (12 surfaces)

| Implementation | 952 rays | 9,845 rays | 99,477 rays | 999,289 rays |
|---|---:|---:|---:|---:|
| Optiland NumPy (shipped) | 9.25 (1.0x) | 11.4 (1.0x) | 137 (1.0x) | 1,803 (1.0x) |
| Numba fused, 1 thread | 0.72 (12.9x) | 7.55 (1.5x) | 75.8 (1.8x) | 784 (2.3x) |
| Rust fused, 1 thread | 0.71 (13.1x) | 7.37 (1.5x) | 74.8 (1.8x) | 761 (2.4x) |
| Numba fused, 4 threads | 0.22 (42.4x) | 2.25 (5.1x) | 21.3 (6.4x) | 210 (8.6x) |
| Rust fused, 4 threads | 0.49 (18.7x) | 2.54 (4.5x) | 19.9 (6.9x) | 205 (8.8x) |

**aspheric_singlet** (3 surfaces)

| Implementation | 952 rays | 9,845 rays | 99,477 rays | 999,289 rays |
|---|---:|---:|---:|---:|
| Optiland NumPy (shipped) | 2.44 (1.0x) | 4.20 (1.0x) | 46.6 (1.0x) | 731 (1.0x) |
| Numba fused, 1 thread | 0.18 (13.4x) | 1.84 (2.3x) | 19.2 (2.4x) | 190 (3.8x) |
| Rust fused, 1 thread | 0.17 (14.7x) | 1.81 (2.3x) | 18.8 (2.5x) | 187 (3.9x) |
| Numba fused, 4 threads | 0.07 (35.1x) | 0.56 (7.5x) | 7.02 (6.6x) | 63.4 (11.5x) |
| Rust fused, 4 threads | 0.13 (18.1x) | 0.71 (5.9x) | 5.97 (7.8x) | 55.3 (13.2x) |

**aspheric_double_gauss** (12 surfaces)

| Implementation | 952 rays | 9,845 rays | 99,477 rays | 999,289 rays |
|---|---:|---:|---:|---:|
| Optiland NumPy (shipped) | 11.7 (1.0x) | 21.4 (1.0x) | 300 (1.0x) | 4,377 (1.0x) |
| Numba fused, 1 thread | 1.21 (9.6x) | 12.5 (1.7x) | 130 (2.3x) | 1,304 (3.4x) |
| Rust fused, 1 thread | 1.22 (9.5x) | 12.4 (1.7x) | 128 (2.3x) | 1,302 (3.4x) |
| Numba fused, 4 threads | 0.44 (26.3x) | 3.32 (6.4x) | 39.0 (7.7x) | 340 (12.9x) |
| Rust fused, 4 threads | 0.79 (14.7x) | 3.51 (6.1x) | 34.7 (8.6x) | 332 (13.2x) |

### 2-D complex FFT

Baseline: `numpy.fft.fft2` (pocketfft), used by Optiland's FFT PSF/MTF.

| Implementation | 256x256 | 512x512 | 1024x1024 | 2048x2048 |
|---|---:|---:|---:|---:|
| numpy.fft.fft2 (shipped) | 0.72 (1.0x) | 4.07 (1.0x) | 23.1 (1.0x) | 143 (1.0x) |
| scipy.fft.fft2, 1 worker | 0.65 (1.1x) | 3.34 (1.2x) | 22.4 (1.0x) | 135 (1.1x) |
| rustfft, 1 thread | 0.68 (1.1x) | 3.88 (1.0x) | 20.5 (1.1x) | 175 (0.8x) |
| scipy.fft.fft2, 4 workers | 0.39 (1.9x) | 2.26 (1.8x) | 7.75 (3.0x) | 39.4 (3.6x) |
| rustfft, 4 threads | 0.58 (1.2x) | 2.14 (1.9x) | 7.70 (3.0x) | 63.4 (2.3x) |

### End to end: Rust kernel swapped into a real Optiland call

| Workload | Optiland as shipped | With Rust kernel | Speedup |
|---|---:|---:|---:|
| ScalarHuygensPSF, 128 rays, 128px image | 3,060 | 886 | 3.5x |
| double_gauss.trace, 205,012 rays | 436 | 149 | 2.9x |
| aspheric_double_gauss.trace, 205,012 rays | 707 | 175 | 4.0x |
