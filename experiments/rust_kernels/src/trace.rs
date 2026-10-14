//! Fused sequential real-ray trace for centered, untilted systems.
//!
//! Each ray is carried through every surface in one pass (ray-major), instead
//! of Optiland's surface-major sequence of whole-array NumPy operations. The
//! per-surface physics mirrors Optiland's NumPy path:
//!
//! * localize: translate by the surface vertex z (`CoordinateSystem`)
//! * distance: `Plane.distance`, `optiland.backend._conic` root selection, or
//!   `NewtonRaphsonGeometry._solve_distance_primal`
//! * propagate: `HomogeneousPropagation.propagate`, incl. absorption
//! * OPD: `opd += t * n_pre`
//! * interact: `RealRays.refract` with `_align_surface_normal`
//! * globalize
//!
//! Scope: no tilts/decenters, no physical apertures, refraction only; this is
//! what the benchmark systems use. The Newton convergence floor is evaluated
//! per ray rather than over the batch maximum; for float64 the user tolerance
//! dominates both unless |t| exceeds ~5e4 mm.

use rayon::prelude::*;

const EPS: f64 = f64::EPSILON;
const DENOM_EPS_MULTIPLIER: f64 = 32.0;
const CONV_EPS_MULTIPLIER: f64 = 8.0;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Plane,
    Conic,
    EvenAsphere,
}

pub struct Surface {
    pub kind: Kind,
    pub vertex_z: f64,
    pub radius: f64,
    pub conic: f64,
    pub coefficients: Vec<f64>,
    pub n_pre: f64,
    pub n_post: f64,
    /// Attenuation per mm in the medium before the surface: 4*pi*k/lambda*1e3.
    pub alpha_pre: f64,
    pub tol: f64,
    pub max_iter: usize,
}

#[derive(Clone, Copy)]
pub struct Ray {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub l: f64,
    pub m: f64,
    pub n: f64,
    pub i: f64,
    pub opd: f64,
}

/// Port of `optiland.backend._conic._conic_candidates` + `_select_distance`
/// without an aperture.
#[inline(always)]
fn conic_distance(r: &Ray, radius: f64, conic: f64) -> f64 {
    let (x, y, z, l, m, n) = (r.x, r.y, r.z, r.l, r.m, r.n);
    let k1 = 1.0 + conic;
    let kz = k1 * z;
    let transverse = x * x + y * y;
    let transverse_dir = l * l + m * m;
    let n2 = n * n;
    let a = transverse_dir + k1 * n2;
    let b = 2.0 * (l * x + m * y + n * (kz - radius));
    let c = transverse + z * (kz - 2.0 * radius);
    let residual_scale = transverse + z.abs() * (kz.abs() + 2.0 * radius.abs());
    let roundoff = 4.0 * EPS;
    let resolved_c = c.abs() > roundoff * residual_scale;
    let d = b * b - 4.0 * a * c;
    let d_ok = d >= 0.0;
    let sqrt_d = if d > 0.0 { d.sqrt() } else { 0.0 };
    let q = -0.5 * (b + sqrt_d.copysign(b));
    let a_ok = a != 0.0 && a.abs() < f64::INFINITY;
    let q_ok = q != 0.0 && q.abs() < f64::INFINITY;
    let t1 = q / if a_ok { a } else { 1.0 };
    let t2 = c / if q_ok { q } else { 1.0 };
    let resolved_step =
        t2 * t2 * (transverse_dir + n2) > roundoff * roundoff * (transverse + z * z);
    let solvable1 = d_ok && a_ok && t1.abs() < f64::INFINITY;
    let solvable2 = d_ok && q_ok && t2.abs() < f64::INFINITY;
    let z1 = z + t1 * n;
    let z2 = z + t2 * n;
    let valid1 = solvable1 && t1 > 0.0 && 1.0 - k1 * z1 / radius >= 0.0;
    let valid2 =
        solvable2 && (resolved_c || resolved_step) && t2 > 0.0 && 1.0 - k1 * z2 / radius >= 0.0;
    let vertex2 = (if solvable2 { z2.abs() } else { f64::INFINITY })
        < (if solvable1 { z1.abs() } else { f64::INFINITY });
    let pick2 = valid2 || (!valid1 && vertex2);
    if solvable1 || solvable2 {
        if pick2 {
            t2
        } else {
            t1
        }
    } else {
        f64::NAN
    }
}

/// `EvenAsphereGeometry.sag`, polynomial evaluated with Horner's rule.
#[inline(always)]
fn asphere_sag(s: &Surface, x: f64, y: f64) -> f64 {
    let r2 = x * x + y * y;
    let base =
        r2 / (s.radius * (1.0 + (1.0 - (1.0 + s.conic) * r2 / (s.radius * s.radius)).sqrt()));
    let mut poly = 0.0;
    for &ci in s.coefficients.iter().rev() {
        poly = poly * r2 + ci;
    }
    base + poly * r2
}

/// Unnormalized slopes (dz/dx, dz/dy) of the even asphere.
#[inline(always)]
fn asphere_slopes(s: &Surface, x: f64, y: f64) -> (f64, f64) {
    let r2 = x * x + y * y;
    let denom = s.radius * (1.0 - (1.0 + s.conic) * r2 / (s.radius * s.radius)).sqrt();
    // d/dr2 of sum C_i r2^(i+1) is sum (i+1) C_i r2^i; dfdx = 2x * that.
    let mut dpoly = 0.0;
    for (i, &ci) in s.coefficients.iter().enumerate().rev() {
        dpoly = dpoly * r2 + (i as f64 + 1.0) * ci;
    }
    (x / denom + 2.0 * x * dpoly, y / denom + 2.0 * y * dpoly)
}

#[inline(always)]
fn conic_slopes(s: &Surface, x: f64, y: f64) -> (f64, f64) {
    let r2 = x * x + y * y;
    let denom = s.radius * (1.0 - (1.0 + s.conic) * r2 / (s.radius * s.radius)).sqrt();
    (x / denom, y / denom)
}

/// Unit normal (nx, ny, nz) from slopes, as in `_normal_components`.
#[inline(always)]
fn normal_from_slopes(fx: f64, fy: f64) -> (f64, f64, f64) {
    let mag = (fx * fx + fy * fy + 1.0).sqrt();
    (fx / mag, fy / mag, -1.0 / mag)
}

#[inline(always)]
fn sign_preserving_floor(value: f64, eps: f64) -> f64 {
    if value.abs() > eps {
        value
    } else if value >= 0.0 {
        eps
    } else {
        -eps
    }
}

/// `NewtonRaphsonGeometry._solve_distance_primal` for a single ray.
#[inline(always)]
fn asphere_distance(s: &Surface, r: &Ray) -> f64 {
    let mut t = conic_distance(r, s.radius, s.conic);
    let tol = s.tol.max(CONV_EPS_MULTIPLIER * EPS * t.abs().max(1.0));
    let residual = |t: f64| asphere_sag(s, r.x + t * r.l, r.y + t * r.m) - (r.z + t * r.n);
    let mut f = residual(t);
    let mut iter = 0;
    // `!(|f| < tol)` keeps NaN rays iterating, exactly like the NumPy mask.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    while !(f.abs() < tol) && iter < s.max_iter {
        let (sx, sy) = asphere_slopes(s, r.x + t * r.l, r.y + t * r.m);
        let (nx, ny, nz) = normal_from_slopes(sx, sy);
        let nz_safe = sign_preserving_floor(nz, DENOM_EPS_MULTIPLIER * EPS);
        let (fx, fy) = (-nx / nz_safe, -ny / nz_safe);
        let df_dt = fx * r.l + fy * r.m - r.n;
        let scale = (fx * r.l).abs() + (fy * r.m).abs() + r.n.abs();
        let tau = DENOM_EPS_MULTIPLIER * EPS * scale.max(1.0);
        let safe = if df_dt.abs() <= tau {
            if df_dt >= 0.0 {
                tau
            } else {
                -tau
            }
        } else {
            df_dt
        };
        t -= f / safe;
        iter += 1;
        f = residual(t);
    }
    t
}

#[inline(always)]
fn trace_surface(s: &Surface, r: &mut Ray) {
    r.z -= s.vertex_z;

    let t = match s.kind {
        Kind::Plane => -r.z / r.n,
        Kind::Conic => conic_distance(r, s.radius, s.conic),
        Kind::EvenAsphere => asphere_distance(s, r),
    };

    r.x += t * r.l;
    r.y += t * r.m;
    r.z += t * r.n;
    if s.alpha_pre > 0.0 {
        r.i *= (-s.alpha_pre * t).exp();
    }
    r.opd += t * s.n_pre;

    let (nx, ny, nz) = match s.kind {
        Kind::Plane => (0.0, 0.0, 1.0),
        Kind::Conic => {
            let (fx, fy) = conic_slopes(s, r.x, r.y);
            normal_from_slopes(fx, fy)
        }
        Kind::EvenAsphere => {
            let (fx, fy) = asphere_slopes(s, r.x, r.y);
            normal_from_slopes(fx, fy)
        }
    };

    // RealRays.refract + _align_surface_normal
    let dot = r.l * nx + r.m * ny + r.n * nz;
    let sgn = if dot > 0.0 {
        1.0
    } else if dot < 0.0 {
        -1.0
    } else if dot == 0.0 {
        0.0
    } else {
        f64::NAN
    };
    let (nx, ny, nz, dot) = (nx * sgn, ny * sgn, nz * sgn, dot.abs());
    let u = s.n_pre / s.n_post;
    let root = (1.0 - u * u * (1.0 - dot * dot)).sqrt();
    r.l = u * r.l + nx * root - u * nx * dot;
    r.m = u * r.m + ny * root - u * ny * dot;
    r.n = u * r.n + nz * root - u * nz * dot;

    r.z += s.vertex_z;
}

/// Traces struct-of-arrays columns (x, y, z, L, M, N, i, opd) and returns
/// new columns. Each output element is written by the thread that traced it,
/// so the copy-in/copy-out is parallel too.
pub fn trace(input: [&[f64]; 8], surfaces: &[Surface], parallel: bool) -> [Vec<f64>; 8] {
    let n = input[0].len();
    let mut out: [Vec<f64>; 8] = std::array::from_fn(|_| vec![0.0; n]);
    let run = |k: usize, o: [&mut f64; 8]| {
        let mut r = Ray {
            x: input[0][k],
            y: input[1][k],
            z: input[2][k],
            l: input[3][k],
            m: input[4][k],
            n: input[5][k],
            i: input[6][k],
            opd: input[7][k],
        };
        for s in surfaces {
            trace_surface(s, &mut r);
        }
        let [x, y, z, l, m, nn, i, opd] = o;
        (*x, *y, *z, *l, *m, *nn, *i, *opd) = (r.x, r.y, r.z, r.l, r.m, r.n, r.i, r.opd);
    };
    let [x, y, z, l, m, nn, i, opd] = &mut out;
    if parallel {
        (
            x.par_iter_mut(),
            y.par_iter_mut(),
            z.par_iter_mut(),
            l.par_iter_mut(),
            m.par_iter_mut(),
            nn.par_iter_mut(),
            i.par_iter_mut(),
            opd.par_iter_mut(),
        )
            .into_par_iter()
            .with_min_len(256)
            .enumerate()
            .for_each(|(k, (a, b, c, d, e, f, g, h))| run(k, [a, b, c, d, e, f, g, h]));
    } else {
        for (k, (((((((a, b), c), d), e), f), g), h)) in x
            .iter_mut()
            .zip(y.iter_mut())
            .zip(z.iter_mut())
            .zip(l.iter_mut())
            .zip(m.iter_mut())
            .zip(nn.iter_mut())
            .zip(i.iter_mut())
            .zip(opd.iter_mut())
            .enumerate()
        {
            run(k, [a, b, c, d, e, f, g, h]);
        }
    }
    out
}
