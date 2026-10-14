//! A flattened lens snapshot and a whole merit evaluation in compiled code.
//!
//! The lens is captured once from an Optiland `Optic` (surface types, radii,
//! conics, thicknesses, refractive index per surface per wavelength, stop,
//! aperture, field scale). Every evaluation then runs without touching
//! Python: paraxial `f2` and entrance pupil, Optiland's paraxial ray aiming
//! for an infinite-conjugate angle field, the real-ray trace, the
//! multi-wavelength RMS spot, and the weighted merit sum.
//!
//! Each step mirrors the Optiland code path it replaces:
//!
//! * `Paraxial.f2`, `Paraxial.EPL` (as the stop imaged into object space)
//! * `AngleField.get_ray_origins` + `ParaxialRayAiming._aim_rays_impl`
//! * `HexagonalDistribution.generate_points`
//! * the real trace from experiment 1 (`_conic_candidates` root selection,
//!   `RealRays.refract`)
//! * `RayOperand.rms_spot_size(wavelength="all")` and
//!   `OptimizationProblem.sum_squared`
//!
//! Scope: centered, untilted planes and conics, refraction only, EPD
//! aperture, angle fields, object at infinity, no vignetting.

use std::f64::consts::PI;

const EPS: f64 = f64::EPSILON;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    Plane,
    Conic,
}

#[derive(Clone, Copy)]
pub enum VarKind {
    Radius,
    Thickness,
}

#[derive(Clone, Copy)]
pub struct Variable {
    pub kind: VarKind,
    pub surface: usize,
    /// Optiland `LinearScaler`: scaled = value * factor + offset.
    pub factor: f64,
    pub offset: f64,
}

#[derive(Clone, Copy)]
pub struct SpotOperand {
    pub hx: f64,
    pub hy: f64,
    pub weight: f64,
    pub target: f64,
}

#[derive(Clone)]
pub struct Lens {
    /// Per surface, index 0 is the object surface.
    pub kind: Vec<Kind>,
    pub radius: Vec<f64>,
    pub conic: Vec<f64>,
    /// Thickness after each surface; entry 0 (object) is unused.
    pub thickness: Vec<f64>,
    /// `index[w][k]`: refractive index after surface k at wavelength w.
    pub index: Vec<Vec<f64>>,
    pub first_z: f64,
    pub stop: usize,
    pub epd: f64,
    /// Largest radial field angle in degrees (`FieldGroup.max_field`).
    pub max_field: f64,
    pub primary: usize,
    /// Hexapolar pupil points used by the RMS operands.
    pub pupil: Vec<(f64, f64)>,
    pub f2_weight: f64,
    pub f2_target: f64,
    pub spots: Vec<SpotOperand>,
    pub variables: Vec<Variable>,
    z: Vec<f64>,
}

#[derive(Clone, Copy, Debug)]
pub struct Ray {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub l: f64,
    pub m: f64,
    pub n: f64,
}

/// `HexagonalDistribution.generate_points(num_rings)`.
pub fn hexapolar(num_rings: usize) -> Vec<(f64, f64)> {
    let mut points = vec![(0.0, 0.0)];
    for i in 0..num_rings {
        let r = (i + 1) as f64 / num_rings as f64;
        let count = 6 * (i + 1);
        for j in 0..count {
            // linspace(0, 2*pi, count + 1)[:-1]
            let theta = 2.0 * PI * j as f64 / count as f64;
            points.push((r * theta.cos(), r * theta.sin()));
        }
    }
    points
}

/// Port of `optiland.backend._conic` root selection without an aperture.
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

/// One surface of `Surface._trace_real` in local coordinates (z already
/// localized): distance, propagate, refract. Returns the distance.
#[inline(always)]
pub fn trace_local(r: &mut Ray, kind: Kind, radius: f64, conic: f64, n1: f64, n2: f64) -> f64 {
    let t = match kind {
        Kind::Plane => -r.z / r.n,
        Kind::Conic => conic_distance(r, radius, conic),
    };
    r.x += t * r.l;
    r.y += t * r.m;
    r.z += t * r.n;

    let (nx, ny, nz) = match kind {
        Kind::Plane => (0.0, 0.0, 1.0),
        Kind::Conic => {
            let r2 = r.x * r.x + r.y * r.y;
            let denom = radius * (1.0 - (1.0 + conic) * r2 / (radius * radius)).sqrt();
            let (fx, fy) = (r.x / denom, r.y / denom);
            let mag = (fx * fx + fy * fy + 1.0).sqrt();
            (fx / mag, fy / mag, -1.0 / mag)
        }
    };
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
    let u = n1 / n2;
    let root = (1.0 - u * u * (1.0 - dot * dot)).sqrt();
    r.l = u * r.l + nx * root - u * nx * dot;
    r.m = u * r.m + ny * root - u * ny * dot;
    r.n = u * r.n + nz * root - u * nz * dot;
    t
}

impl Lens {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        kind: Vec<Kind>,
        radius: Vec<f64>,
        conic: Vec<f64>,
        thickness: Vec<f64>,
        index: Vec<Vec<f64>>,
        first_z: f64,
        stop: usize,
        epd: f64,
        max_field: f64,
        primary: usize,
        num_rings: usize,
    ) -> Self {
        let mut lens = Lens {
            kind,
            radius,
            conic,
            thickness,
            index,
            first_z,
            stop,
            epd,
            max_field,
            primary,
            pupil: hexapolar(num_rings),
            f2_weight: 0.0,
            f2_target: 0.0,
            spots: Vec::new(),
            variables: Vec::new(),
            z: Vec::new(),
        };
        lens.refresh_positions();
        lens
    }

    pub fn num_surfaces(&self) -> usize {
        self.kind.len()
    }

    /// Surface vertex z positions from the thicknesses (`SurfaceGroup.positions`).
    pub fn refresh_positions(&mut self) {
        let s = self.num_surfaces();
        self.z = vec![f64::NEG_INFINITY; s];
        self.z[1] = self.first_z;
        for k in 1..s - 1 {
            self.z[k + 1] = self.z[k] + self.thickness[k];
        }
    }

    pub fn positions(&self) -> &[f64] {
        &self.z
    }

    /// Paraxial y-u trace from just before surface 1 through surfaces 1..=last,
    /// refracting at each; returns (y, u) after surface `last`. When
    /// `refract_last` is false the height is taken on arrival at `last`.
    fn paraxial(
        &self,
        w: usize,
        mut y: f64,
        mut u: f64,
        last: usize,
        refract_last: bool,
    ) -> (f64, f64) {
        let n = &self.index[w];
        for k in 1..=last {
            if k > 1 {
                y += u * self.thickness[k - 1];
            }
            if k == last && !refract_last {
                break;
            }
            let c = match self.kind[k] {
                Kind::Plane => 0.0,
                Kind::Conic => 1.0 / self.radius[k],
            };
            u = (n[k - 1] * u - y * (n[k] - n[k - 1]) * c) / n[k];
        }
        (y, u)
    }

    /// `Paraxial.f2`: unit-height ray parallel to the axis.
    pub fn f2(&self) -> f64 {
        let (_, u) = self.paraxial(self.primary, 1.0, 0.0, self.num_surfaces() - 1, true);
        -1.0 / u
    }

    /// `Paraxial.EPL`: axial position of the stop's image in object space,
    /// relative to surface 1. Paraxial optics is linear, so the height at
    /// the stop is A*h + B*u for a ray (h, u) at surface 1, and the chief
    /// ray (stop height zero) crosses the axis at z = B / A.
    pub fn epl(&self) -> f64 {
        if self.stop == 1 {
            return 0.0;
        }
        let (a, _) = self.paraxial(self.primary, 1.0, 0.0, self.stop, false);
        let (b, _) = self.paraxial(self.primary, 0.0, 1.0, self.stop, false);
        b / a
    }

    /// Launch state for normalized field (hx, hy) and pupil (px, py):
    /// `AngleField.get_ray_origins` followed by paraxial ray aiming.
    pub fn launch(&self, epl: f64, hx: f64, hy: f64, px: f64, py: f64) -> Ray {
        let fx = self.max_field * hx;
        let fy = self.max_field * hy;
        let s = self.num_surfaces();
        let min_rel = (1..s - 1)
            .map(|k| self.z[k] - self.z[1])
            .fold(f64::INFINITY, f64::min);
        let d = (self.epd - min_rel) + epl;
        let theta = (fx * fx + fy * fy).sqrt();
        let sign = if theta.to_radians().cos() < 0.0 {
            -1.0
        } else {
            1.0
        };
        let z_pupil = epl + self.z[1];
        let x = -sign * fx.to_radians().tan() * d;
        let y = -sign * fy.to_radians().tan() * d;
        let (x1, y1) = (px * self.epd / 2.0, py * self.epd / 2.0);
        let (x0, y0, z0) = (x1 + x, y1 + y, z_pupil - sign * d);
        let (dx, dy, dz) = (x1 - x0, y1 - y0, z_pupil - z0);
        let mag = (dx * dx + dy * dy + dz * dz).sqrt();
        if mag < 1e-9 {
            return Ray {
                x: x0,
                y: y0,
                z: z0,
                l: 0.0,
                m: 0.0,
                n: 1.0,
            };
        }
        Ray {
            x: x0,
            y: y0,
            z: z0,
            l: dx / mag,
            m: dy / mag,
            n: dz / mag,
        }
    }

    /// Real trace of one ray through surfaces 1..N at wavelength `w`.
    pub fn trace(&self, w: usize, mut r: Ray) -> Ray {
        let n = &self.index[w];
        for k in 1..self.num_surfaces() {
            r.z -= self.z[k];
            trace_local(
                &mut r,
                self.kind[k],
                self.radius[k],
                self.conic[k],
                n[k - 1],
                n[k],
            );
            r.z += self.z[k];
        }
        r
    }

    /// `rms_spot_size(wavelength="all", nan_policy="propagate")` on the image.
    pub fn rms_spot(&self, epl: f64, hx: f64, hy: f64) -> f64 {
        let waves = self.index.len();
        let np = self.pupil.len();
        let mut xs = vec![0.0; waves * np];
        let mut ys = vec![0.0; waves * np];
        for w in 0..waves {
            for (j, &(px, py)) in self.pupil.iter().enumerate() {
                let r = self.trace(w, self.launch(epl, hx, hy, px, py));
                xs[w * np + j] = r.x;
                ys[w * np + j] = r.y;
            }
        }
        if xs.iter().chain(&ys).any(|v| v.is_nan()) {
            return f64::NAN;
        }
        let p = self.primary * np;
        let cx = xs[p..p + np].iter().sum::<f64>() / np as f64;
        let cy = ys[p..p + np].iter().sum::<f64>() / np as f64;
        let r2: f64 = xs
            .iter()
            .zip(&ys)
            .map(|(x, y)| (x - cx) * (x - cx) + (y - cy) * (y - cy))
            .sum();
        (r2 / xs.len() as f64).sqrt()
    }

    pub fn apply(&mut self, x: &[f64]) {
        for (v, &xi) in self.variables.iter().zip(x) {
            let value = (xi - v.offset) / v.factor;
            match v.kind {
                VarKind::Radius => self.radius[v.surface] = value,
                VarKind::Thickness => self.thickness[v.surface] = value,
            }
        }
        self.refresh_positions();
    }

    /// `OptimizationProblem.residual_vector()`: weighted deltas, f2 first.
    pub fn residuals(&self) -> Vec<f64> {
        let epl = self.epl();
        let mut out = Vec::with_capacity(1 + self.spots.len());
        out.push(self.f2_weight * (self.f2() - self.f2_target));
        for op in &self.spots {
            out.push(op.weight * (self.rms_spot(epl, op.hx, op.hy) - op.target));
        }
        out
    }

    /// `OptimizationProblem.sum_squared()` for f2 + RMS-spot operands.
    pub fn merit(&self) -> f64 {
        self.residuals().iter().map(|r| r * r).sum()
    }

    /// Residuals at scaled variable vector `x`; the snapshot is unchanged.
    pub fn residuals_at(&self, x: &[f64]) -> Vec<f64> {
        let mut work = self.clone();
        work.apply(x);
        work.residuals()
    }

    /// Merit at variable vector `x` (scaled, as the optimizer passes it).
    pub fn merit_at(&self, x: &[f64]) -> f64 {
        let mut work = self.clone();
        work.apply(x);
        work.merit()
    }
}
