//! Huygens-Fresnel PSF summation.
//!
//! Three variants, so language effects can be separated from algorithmic ones:
//!
//! * `literal`: line-for-line port of Optiland's Numba kernel
//!   (`NumbaSummation._huygens_fresnel_summation`), including the per-pair
//!   pupil phase `exp(-i k opd_j)`.
//! * `hoisted`: the pupil phase and amplitude depend only on `j`, so they are
//!   precomputed once as a complex weight. One `sin_cos` per pair instead of
//!   two. Still scalar libm.
//! * `simd`: `hoisted` plus the branch-free `fastmath::sin_cos` and explicit
//!   8-lane accumulators, which lets LLVM emit AVX-512.
//!
//! Every variant parallelizes over image points with rayon when `parallel`.

use crate::fastmath;
use num_complex::Complex64;
use rayon::prelude::*;

pub struct Pupil<'a> {
    pub x: &'a [f64],
    pub y: &'a [f64],
    pub z: &'a [f64],
    pub amp: &'a [f64],
    pub opd: &'a [f64],
}

fn literal_point(px: f64, py: f64, pz: f64, pupil: &Pupil, k: f64, rp: f64) -> f64 {
    let mut sum = Complex64::new(0.0, 0.0);
    for j in 0..pupil.x.len() {
        let (u, v, w) = (pupil.x[j], pupil.y[j], pupil.z[j]);
        let (dx, dy, dz) = (px - u, py - v, pz - w);
        let r = (dx * dx + dy * dy + dz * dz).sqrt();
        let (s, c) = (k * r).sin_cos();
        let wave = Complex64::new(c, s) / r;
        let dot = dx * (u / rp) + dy * (v / rp) + dz * (w / rp);
        let obliquity = 0.5 * (1.0 + dot / r);
        let (ps, pc) = (-k * pupil.opd[j]).sin_cos();
        sum += pupil.amp[j] * Complex64::new(pc, ps) * wave * obliquity;
    }
    sum.norm_sqr()
}

/// Pupil data rearranged for the hoisted kernels: unit normals and the
/// complex weight `amp_j * exp(-i k opd_j)`, zero-padded to a lane multiple.
struct Prepared {
    x: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
    nx: Vec<f64>,
    ny: Vec<f64>,
    nz: Vec<f64>,
    wr: Vec<f64>,
    wi: Vec<f64>,
}

const LANES: usize = 8;

impl Prepared {
    fn new(pupil: &Pupil, k: f64, rp: f64) -> Self {
        let n = pupil.x.len();
        let padded = n.div_ceil(LANES) * LANES;
        // Padding lanes sit at a real pupil point with zero weight, so R stays
        // finite and they contribute exactly zero.
        let pad = |v: &[f64]| {
            let mut out = v.to_vec();
            out.resize(padded, v.last().copied().unwrap_or(1.0));
            out
        };
        let (x, y, z) = (pad(pupil.x), pad(pupil.y), pad(pupil.z));
        let nx = x.iter().map(|u| u / rp).collect();
        let ny = y.iter().map(|v| v / rp).collect();
        let nz = z.iter().map(|w| w / rp).collect();
        let mut wr = vec![0.0; padded];
        let mut wi = vec![0.0; padded];
        for j in 0..n {
            let (s, c) = (-k * pupil.opd[j]).sin_cos();
            wr[j] = pupil.amp[j] * c;
            wi[j] = pupil.amp[j] * s;
        }
        Prepared {
            x,
            y,
            z,
            nx,
            ny,
            nz,
            wr,
            wi,
        }
    }
}

fn hoisted_point(px: f64, py: f64, pz: f64, p: &Prepared, k: f64) -> f64 {
    let (mut re, mut im) = (0.0, 0.0);
    for j in 0..p.x.len() {
        let (dx, dy, dz) = (px - p.x[j], py - p.y[j], pz - p.z[j]);
        let r = (dx * dx + dy * dy + dz * dz).sqrt();
        let inv_r = 1.0 / r;
        let dot = dx * p.nx[j] + dy * p.ny[j] + dz * p.nz[j];
        let scale = 0.5 * (1.0 + dot * inv_r) * inv_r;
        let (s, c) = (k * r).sin_cos();
        re += scale * (p.wr[j] * c - p.wi[j] * s);
        im += scale * (p.wr[j] * s + p.wi[j] * c);
    }
    re * re + im * im
}

fn simd_point(px: f64, py: f64, pz: f64, p: &Prepared, k: f64) -> f64 {
    let mut re = [0.0f64; LANES];
    let mut im = [0.0f64; LANES];
    let chunks = p.x.len() / LANES;
    for chunk in 0..chunks {
        let base = chunk * LANES;
        for l in 0..LANES {
            let j = base + l;
            let (dx, dy, dz) = (px - p.x[j], py - p.y[j], pz - p.z[j]);
            let r2 = dx.mul_add(dx, dy.mul_add(dy, dz * dz));
            let r = r2.sqrt();
            let inv_r = 1.0 / r;
            let dot = dx.mul_add(p.nx[j], dy.mul_add(p.ny[j], dz * p.nz[j]));
            let scale = 0.5 * dot.mul_add(inv_r, 1.0) * inv_r;
            let (s, c) = fastmath::sin_cos(k * r);
            re[l] = scale.mul_add(p.wr[j].mul_add(c, -p.wi[j] * s), re[l]);
            im[l] = scale.mul_add(p.wr[j].mul_add(s, p.wi[j] * c), im[l]);
        }
    }
    let re: f64 = re.iter().sum();
    let im: f64 = im.iter().sum();
    re * re + im * im
}

#[derive(Clone, Copy)]
pub enum Variant {
    Literal,
    Hoisted,
    Simd,
}

/// `image_*` are flattened image grids; returns |field|^2 in the same order.
pub fn psf(
    image: (&[f64], &[f64], &[f64]),
    pupil: &Pupil,
    wavelength: f64,
    rp: f64,
    variant: Variant,
    parallel: bool,
) -> Vec<f64> {
    let k = 2.0 * std::f64::consts::PI / wavelength;
    let (ix, iy, iz) = image;
    let prepared = match variant {
        Variant::Literal => None,
        _ => Some(Prepared::new(pupil, k, rp)),
    };
    let eval = |i: usize| match variant {
        Variant::Literal => literal_point(ix[i], iy[i], iz[i], pupil, k, rp),
        Variant::Hoisted => hoisted_point(ix[i], iy[i], iz[i], prepared.as_ref().unwrap(), k),
        Variant::Simd => simd_point(ix[i], iy[i], iz[i], prepared.as_ref().unwrap(), k),
    };
    if parallel {
        (0..ix.len()).into_par_iter().map(eval).collect()
    } else {
        (0..ix.len()).map(eval).collect()
    }
}
