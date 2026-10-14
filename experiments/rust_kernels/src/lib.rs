//! Python bindings for the experimental Rust kernels.
//!
//! All entry points release the GIL while computing. `parallel=True` uses the
//! global rayon pool (sized by `RAYON_NUM_THREADS`, default: all cores).

mod fastmath;
mod fft;
mod huygens;
mod trace;

use numpy::ndarray::Array2;
use numpy::{
    Complex64, IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

type F64Array<'py> = Bound<'py, PyArray1<f64>>;

#[pyfunction]
#[pyo3(signature = (
    image_x, image_y, image_z, pupil_x, pupil_y, pupil_z, pupil_amp, pupil_opd,
    wavelength, rp, reference_center, variant = "simd", parallel = true
))]
#[allow(clippy::too_many_arguments)]
fn huygens_psf<'py>(
    py: Python<'py>,
    image_x: PyReadonlyArray2<'py, f64>,
    image_y: PyReadonlyArray2<'py, f64>,
    image_z: PyReadonlyArray2<'py, f64>,
    pupil_x: PyReadonlyArray1<'py, f64>,
    pupil_y: PyReadonlyArray1<'py, f64>,
    pupil_z: PyReadonlyArray1<'py, f64>,
    pupil_amp: PyReadonlyArray1<'py, f64>,
    pupil_opd: PyReadonlyArray1<'py, f64>,
    wavelength: f64,
    rp: f64,
    reference_center: (f64, f64, f64),
    variant: &str,
    parallel: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let variant = match variant {
        "literal" => huygens::Variant::Literal,
        "hoisted" => huygens::Variant::Hoisted,
        "simd" => huygens::Variant::Simd,
        other => return Err(PyValueError::new_err(format!("unknown variant {other:?}"))),
    };
    let shape = image_x.shape().to_vec();
    if image_y.shape() != shape.as_slice() || image_z.shape() != shape.as_slice() {
        return Err(PyValueError::new_err("image arrays must share one shape"));
    }
    let image = (
        image_x.as_slice()?,
        image_y.as_slice()?,
        image_z.as_slice()?,
    );
    let pupil = huygens::Pupil {
        x: pupil_x.as_slice()?,
        y: pupil_y.as_slice()?,
        z: pupil_z.as_slice()?,
        amp: pupil_amp.as_slice()?,
        opd: pupil_opd.as_slice()?,
        center: reference_center,
    };
    let np = pupil.x.len();
    if [pupil.y, pupil.z, pupil.amp, pupil.opd]
        .iter()
        .any(|a| a.len() != np)
    {
        return Err(PyValueError::new_err("pupil arrays must have equal length"));
    }
    let out = py.detach(|| huygens::psf(image, &pupil, wavelength, rp, variant, parallel));
    let out = Array2::from_shape_vec((shape[0], shape[1]), out)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(out.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (
    rays, kinds, vertex_z, radius, conic, coefficients, n_pre, n_post, alpha_pre,
    tol = 1e-10, max_iter = 100, parallel = true
))]
#[allow(clippy::too_many_arguments)]
fn trace_sequential<'py>(
    py: Python<'py>,
    rays: [PyReadonlyArray1<'py, f64>; 8],
    kinds: PyReadonlyArray1<'py, i64>,
    vertex_z: PyReadonlyArray1<'py, f64>,
    radius: PyReadonlyArray1<'py, f64>,
    conic: PyReadonlyArray1<'py, f64>,
    coefficients: PyReadonlyArray2<'py, f64>,
    n_pre: PyReadonlyArray1<'py, f64>,
    n_post: PyReadonlyArray1<'py, f64>,
    alpha_pre: PyReadonlyArray1<'py, f64>,
    tol: f64,
    max_iter: usize,
    parallel: bool,
) -> PyResult<[F64Array<'py>; 8]> {
    let coefficients = coefficients.as_array();
    let nsurf = kinds.len();
    let per_surface = [&vertex_z, &radius, &conic, &n_pre, &n_post, &alpha_pre];
    if per_surface.iter().any(|a| a.len() != nsurf) || coefficients.nrows() != nsurf {
        return Err(PyValueError::new_err(
            "surface arrays must have one row per surface",
        ));
    }
    let mut surfaces = Vec::with_capacity(nsurf);
    for (s, &kind) in kinds.as_slice()?.iter().enumerate() {
        let kind = match kind {
            0 => trace::Kind::Plane,
            1 => trace::Kind::Conic,
            2 => trace::Kind::EvenAsphere,
            other => return Err(PyValueError::new_err(format!("unknown kind {other}"))),
        };
        // Trailing zero coefficients are padding from the 2-D layout.
        let mut coeffs = coefficients.row(s).to_vec();
        while coeffs.last() == Some(&0.0) {
            coeffs.pop();
        }
        surfaces.push(trace::Surface {
            kind,
            vertex_z: vertex_z.as_slice()?[s],
            radius: radius.as_slice()?[s],
            conic: conic.as_slice()?[s],
            coefficients: coeffs,
            n_pre: n_pre.as_slice()?[s],
            n_post: n_post.as_slice()?[s],
            alpha_pre: alpha_pre.as_slice()?[s],
            tol,
            max_iter,
        });
    }

    let cols: Vec<&[f64]> = rays
        .iter()
        .map(|a| a.as_slice())
        .collect::<Result<_, _>>()?;
    let n = cols[0].len();
    if cols.iter().any(|c| c.len() != n) {
        return Err(PyValueError::new_err("ray arrays must have equal length"));
    }
    let cols: [&[f64]; 8] = cols.try_into().expect("eight ray columns");
    let out = py.detach(|| trace::trace(cols, &surfaces, parallel));
    Ok(out.map(|v| v.into_pyarray(py)))
}

#[pyfunction]
#[pyo3(signature = (data, parallel = true))]
fn fft2<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, Complex64>,
    parallel: bool,
) -> PyResult<Bound<'py, PyArray2<Complex64>>> {
    let shape = data.shape().to_vec();
    let input = data.as_slice()?;
    let out = py.detach(|| fft::fft2(input, shape[0], shape[1], parallel));
    let out = Array2::from_shape_vec((shape[0], shape[1]), out)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(out.into_pyarray(py))
}

#[pyfunction]
fn num_threads() -> usize {
    rayon::current_num_threads()
}

#[pymodule]
fn optiland_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(huygens_psf, m)?)?;
    m.add_function(wrap_pyfunction!(trace_sequential, m)?)?;
    m.add_function(wrap_pyfunction!(fft2, m)?)?;
    m.add_function(wrap_pyfunction!(num_threads, m)?)?;
    Ok(())
}
