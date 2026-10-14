//! 2-D complex FFT with rustfft, for comparison with `numpy.fft.fft2`.
//!
//! Row FFTs, a blocked transpose, row FFTs again, and a transpose back. Plans
//! are cached across calls (as pocketfft caches its own). Every pass that
//! touches the full array (copy-in, FFTs, transposes) runs on the rayon pool
//! when `parallel`, so no single-threaded memory sweep remains.

use num_complex::Complex64;
use rayon::prelude::*;
use rustfft::{Fft, FftPlanner};
use std::mem::MaybeUninit;
use std::sync::{Arc, Mutex, OnceLock};

const ZERO: Complex64 = Complex64::new(0.0, 0.0);
/// Destination rows produced together; keeps source reads in short runs.
const BLOCK: usize = 16;

fn plan(len: usize) -> Arc<dyn Fft<f64>> {
    static PLANNER: OnceLock<Mutex<FftPlanner<f64>>> = OnceLock::new();
    PLANNER
        .get_or_init(|| Mutex::new(FftPlanner::new()))
        .lock()
        .unwrap()
        .plan_fft_forward(len)
}

/// Fills a fresh `len`-element vector chunk by chunk, possibly in parallel.
/// `fill(chunk_index, chunk)` must write every element of its chunk.
fn build<F>(len: usize, chunk: usize, parallel: bool, fill: F) -> Vec<Complex64>
where
    F: Fn(usize, &mut [MaybeUninit<Complex64>]) + Sync,
{
    let mut out = Vec::with_capacity(len);
    let spare = &mut out.spare_capacity_mut()[..len];
    if parallel {
        spare
            .par_chunks_mut(chunk)
            .enumerate()
            .for_each(|(i, c)| fill(i, c));
    } else {
        spare
            .chunks_mut(chunk)
            .enumerate()
            .for_each(|(i, c)| fill(i, c));
    }
    // SAFETY: every caller's `fill` writes each element of its chunk, and the
    // chunks cover `0..len`.
    unsafe { out.set_len(len) };
    out
}

/// Copies `src` (`rows x cols`) and FFTs each row in the same pass.
fn copy_and_fft_rows(src: &[Complex64], cols: usize, parallel: bool) -> Vec<Complex64> {
    let fft = plan(cols);
    let mut out = build(src.len(), cols, parallel, |r, row| {
        for (dst, &v) in row.iter_mut().zip(&src[r * cols..(r + 1) * cols]) {
            dst.write(v);
        }
    });
    fft_rows(&mut out, &fft, parallel);
    out
}

fn fft_rows(data: &mut [Complex64], fft: &Arc<dyn Fft<f64>>, parallel: bool) {
    let len = fft.len();
    let scratch_len = fft.get_inplace_scratch_len();
    if parallel {
        data.par_chunks_mut(len).with_min_len(4).for_each_init(
            || vec![ZERO; scratch_len],
            |scratch, row| fft.process_with_scratch(row, scratch),
        );
    } else {
        // rustfft processes consecutive rows itself when given the whole buffer.
        fft.process_with_scratch(data, &mut vec![ZERO; scratch_len]);
    }
}

/// `src` is `rows x cols`; returns the `cols x rows` transpose.
fn transpose(src: &[Complex64], rows: usize, cols: usize, parallel: bool) -> Vec<Complex64> {
    build(src.len(), BLOCK * rows, parallel, |block, out| {
        let c0 = block * BLOCK;
        let width = out.len() / rows;
        for r in 0..rows {
            let run = &src[r * cols + c0..r * cols + c0 + width];
            for (i, &v) in run.iter().enumerate() {
                out[i * rows + r].write(v);
            }
        }
    })
}

/// Forward 2-D FFT of a row-major `rows x cols` array.
pub fn fft2(data: &[Complex64], rows: usize, cols: usize, parallel: bool) -> Vec<Complex64> {
    let rows_done = copy_and_fft_rows(data, cols, parallel);
    let mut t = transpose(&rows_done, rows, cols, parallel);
    drop(rows_done);
    fft_rows(&mut t, &plan(rows), parallel);
    transpose(&t, cols, rows, parallel)
}
