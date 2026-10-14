//! Python bindings for the small-operations experiment.
//!
//! Three tiers, from finest to coarsest granularity:
//!
//! 1. Boundary probes (`noop`, `add3`, `echo_arrays8`): the fixed cost of a
//!    Python -> Rust call, with scalars and with small NumPy arrays.
//! 2. `trace_surface`: one surface of Optiland's real-ray kernel
//!    (`Surface._trace_real`), a drop-in leaf replacement that leaves the
//!    rest of Optiland's Python orchestration in place.
//! 3. `CompiledLens`: a flattened lens snapshot that evaluates paraxial
//!    quantities, ray aiming, traces and the whole merit function per call.

mod lens;

use lens::{Kind, Lens, Ray, SpotOperand, VarKind, Variable};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

type F64Array<'py> = Bound<'py, PyArray1<f64>>;

// ---------------------------------------------------------------------------
// Tier 1: boundary cost
// ---------------------------------------------------------------------------

#[pyfunction]
fn noop() {}

#[pyfunction]
fn add3(a: f64, b: f64, c: f64) -> f64 {
    a + b + c
}

/// Borrow eight 1-D arrays and return eight new arrays (copies).
#[pyfunction]
fn echo_arrays8<'py>(
    py: Python<'py>,
    arrays: [PyReadonlyArray1<'py, f64>; 8],
) -> PyResult<[F64Array<'py>; 8]> {
    let mut out: [Vec<f64>; 8] = Default::default();
    for (dst, src) in out.iter_mut().zip(&arrays) {
        *dst = src.as_slice()?.to_vec();
    }
    Ok(out.map(|v| v.into_pyarray(py)))
}

// ---------------------------------------------------------------------------
// Tier 2: one surface of the real-ray kernel
// ---------------------------------------------------------------------------

fn kind_from(code: i64) -> PyResult<Kind> {
    match code {
        0 => Ok(Kind::Plane),
        1 => Ok(Kind::Conic),
        other => Err(PyValueError::new_err(format!(
            "unknown surface kind {other}"
        ))),
    }
}

/// `Surface._trace_real` for a plane or conic in local coordinates:
/// distance, propagate (with absorption `alpha` per mm), OPD, refract.
/// `rays` is (x, y, z, L, M, N, i, opd); returns the same eight columns.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn trace_surface<'py>(
    py: Python<'py>,
    rays: [PyReadonlyArray1<'py, f64>; 8],
    kind: i64,
    radius: f64,
    conic: f64,
    n1: f64,
    n2: f64,
    alpha: f64,
) -> PyResult<[F64Array<'py>; 8]> {
    let kind = kind_from(kind)?;
    let cols: Vec<&[f64]> = rays
        .iter()
        .map(|a| a.as_slice())
        .collect::<Result<_, _>>()?;
    let n = cols[0].len();
    if cols.iter().any(|c| c.len() != n) {
        return Err(PyValueError::new_err("ray arrays must have equal length"));
    }
    let mut out: [Vec<f64>; 8] = std::array::from_fn(|_| Vec::with_capacity(n));
    #[allow(clippy::needless_range_loop)] // k indexes all eight columns
    for k in 0..n {
        let mut r = Ray {
            x: cols[0][k],
            y: cols[1][k],
            z: cols[2][k],
            l: cols[3][k],
            m: cols[4][k],
            n: cols[5][k],
        };
        let t = lens::trace_local(&mut r, kind, radius, conic, n1, n2);
        let mut intensity = cols[6][k];
        if alpha > 0.0 {
            intensity *= (-alpha * t).exp();
        }
        for (col, v) in
            out.iter_mut()
                .zip([r.x, r.y, r.z, r.l, r.m, r.n, intensity, cols[7][k] + t * n1])
        {
            col.push(v);
        }
    }
    Ok(out.map(|v| v.into_pyarray(py)))
}

// ---------------------------------------------------------------------------
// Tier 3: compiled lens
// ---------------------------------------------------------------------------

#[pyclass]
struct CompiledLens {
    lens: Lens,
}

#[pymethods]
impl CompiledLens {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        kinds: Vec<i64>,
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
    ) -> PyResult<Self> {
        let s = kinds.len();
        if s < 3
            || [radius.len(), conic.len(), thickness.len()]
                .iter()
                .any(|&l| l != s)
        {
            return Err(PyValueError::new_err(
                "per-surface lists must share a length >= 3",
            ));
        }
        if index.is_empty() || index.iter().any(|row| row.len() != s) {
            return Err(PyValueError::new_err("index must be [wavelength][surface]"));
        }
        if stop == 0 || stop >= s || primary >= index.len() {
            return Err(PyValueError::new_err("stop or primary index out of range"));
        }
        let kinds = kinds
            .into_iter()
            .map(kind_from)
            .collect::<PyResult<Vec<_>>>()?;
        Ok(CompiledLens {
            lens: Lens::new(
                kinds, radius, conic, thickness, index, first_z, stop, epd, max_field, primary,
                num_rings,
            ),
        })
    }

    /// Operands: f2 (weight, target) and RMS spots [(hx, hy, weight, target)].
    fn set_operands(&mut self, f2_weight: f64, f2_target: f64, spots: Vec<(f64, f64, f64, f64)>) {
        self.lens.f2_weight = f2_weight;
        self.lens.f2_target = f2_target;
        self.lens.spots = spots
            .into_iter()
            .map(|(hx, hy, weight, target)| SpotOperand {
                hx,
                hy,
                weight,
                target,
            })
            .collect();
    }

    /// Variables: [("radius" | "thickness", surface, factor, offset)].
    fn set_variables(&mut self, variables: Vec<(String, usize, f64, f64)>) -> PyResult<()> {
        let s = self.lens.num_surfaces();
        let mut out = Vec::with_capacity(variables.len());
        for (kind, surface, factor, offset) in variables {
            let kind = match kind.as_str() {
                "radius" => VarKind::Radius,
                "thickness" => VarKind::Thickness,
                other => return Err(PyValueError::new_err(format!("unknown variable {other}"))),
            };
            if surface == 0 || surface >= s {
                return Err(PyValueError::new_err("variable surface out of range"));
            }
            out.push(Variable {
                kind,
                surface,
                factor,
                offset,
            });
        }
        self.lens.variables = out;
        Ok(())
    }

    fn f2(&self) -> f64 {
        self.lens.f2()
    }

    fn epl(&self) -> f64 {
        self.lens.epl()
    }

    fn positions(&self) -> Vec<f64> {
        self.lens.positions().to_vec()
    }

    /// Final (x, y, z, L, M, N) of one ray, like `Optic.trace_generic`.
    fn trace_generic(
        &self,
        hx: f64,
        hy: f64,
        px: f64,
        py: f64,
        wavelength_index: usize,
    ) -> PyResult<(f64, f64, f64, f64, f64, f64)> {
        self.check_wavelength(wavelength_index)?;
        let r = self.lens.trace(
            wavelength_index,
            self.lens.launch(self.lens.epl(), hx, hy, px, py),
        );
        Ok((r.x, r.y, r.z, r.l, r.m, r.n))
    }

    /// Image-plane (x, y) of every pupil point at one wavelength.
    fn trace_pupil(
        &self,
        hx: f64,
        hy: f64,
        wavelength_index: usize,
    ) -> PyResult<(Vec<f64>, Vec<f64>)> {
        self.check_wavelength(wavelength_index)?;
        let epl = self.lens.epl();
        Ok(self
            .lens
            .pupil
            .iter()
            .map(|&(px, py)| {
                let r = self
                    .lens
                    .trace(wavelength_index, self.lens.launch(epl, hx, hy, px, py));
                (r.x, r.y)
            })
            .unzip())
    }

    fn rms_spot(&self, hx: f64, hy: f64) -> f64 {
        self.lens.rms_spot(self.lens.epl(), hx, hy)
    }

    /// Merit of the current state.
    fn merit(&self) -> f64 {
        self.lens.merit()
    }

    /// Merit at scaled variable vector `x`; the snapshot itself is unchanged.
    fn merit_at(&self, x: Vec<f64>) -> PyResult<f64> {
        self.check_x(x.len())?;
        Ok(self.lens.merit_at(&x))
    }

    /// Weighted operand deltas at `x` (`OptimizationProblem.residual_vector`).
    fn residuals_at(&self, x: Vec<f64>) -> PyResult<Vec<f64>> {
        self.check_x(x.len())?;
        Ok(self.lens.residuals_at(&x))
    }

    /// Residual vectors for each row of `xs`, as a (rows, operands) array.
    #[pyo3(signature = (xs, parallel = true))]
    fn residuals_many<'py>(
        &self,
        py: Python<'py>,
        xs: PyReadonlyArray2<'py, f64>,
        parallel: bool,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let shape = xs.shape();
        self.check_x(shape[1])?;
        let rows = xs.as_slice()?;
        let width = shape[1];
        let lens = &self.lens;
        let m = 1 + lens.spots.len();
        let out: Vec<f64> = py.detach(|| {
            let per_row: Vec<Vec<f64>> = if parallel {
                rows.par_chunks(width)
                    .map(|x| lens.residuals_at(x))
                    .collect()
            } else {
                rows.chunks(width).map(|x| lens.residuals_at(x)).collect()
            };
            per_row.concat()
        });
        let out = numpy::ndarray::Array2::from_shape_vec((shape[0], m), out)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(out.into_pyarray(py))
    }

    /// Apply scaled variables to the snapshot (persisting).
    fn apply(&mut self, x: Vec<f64>) -> PyResult<()> {
        self.check_x(x.len())?;
        self.lens.apply(&x);
        Ok(())
    }

    /// Merit for each row of `xs` (e.g. all finite-difference probes at once).
    #[pyo3(signature = (xs, parallel = true))]
    fn merit_many<'py>(
        &self,
        py: Python<'py>,
        xs: PyReadonlyArray2<'py, f64>,
        parallel: bool,
    ) -> PyResult<F64Array<'py>> {
        let shape = xs.shape();
        self.check_x(shape[1])?;
        let rows = xs.as_slice()?;
        let width = shape[1];
        let lens = &self.lens;
        let out: Vec<f64> = py.detach(|| {
            if parallel {
                rows.par_chunks(width).map(|x| lens.merit_at(x)).collect()
            } else {
                rows.chunks(width).map(|x| lens.merit_at(x)).collect()
            }
        });
        Ok(out.into_pyarray(py))
    }
}

impl CompiledLens {
    fn check_wavelength(&self, w: usize) -> PyResult<()> {
        if w >= self.lens.index.len() {
            return Err(PyValueError::new_err("wavelength index out of range"));
        }
        Ok(())
    }

    fn check_x(&self, len: usize) -> PyResult<()> {
        if len != self.lens.variables.len() {
            return Err(PyValueError::new_err(format!(
                "expected {} variables, got {len}",
                self.lens.variables.len()
            )));
        }
        Ok(())
    }
}

#[pymodule]
fn optiland_small(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(noop, m)?)?;
    m.add_function(wrap_pyfunction!(add3, m)?)?;
    m.add_function(wrap_pyfunction!(echo_arrays8, m)?)?;
    m.add_function(wrap_pyfunction!(trace_surface, m)?)?;
    m.add_class::<CompiledLens>()?;
    Ok(())
}
