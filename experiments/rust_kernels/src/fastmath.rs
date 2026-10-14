//! Branch-free `sin_cos` that LLVM can auto-vectorize.
//!
//! `f64::sin_cos` lowers to a scalar libm call, which blocks SIMD in the
//! Huygens inner loop. This version uses Cody-Waite reduction by pi/2 with a
//! three-part constant applied through FMA, followed by the fdlibm minimax
//! kernels on [-pi/4, pi/4]. Quadrant fix-up is done with selects and sign-bit
//! XORs so every lane executes the same instructions.
//!
//! Accuracy: about 1 ulp for |x| < 2^30. The Huygens phase k*R is ~1e6 rad,
//! where the representation error of k*R itself (ulp ~1e-10) dominates.

// Polynomial coefficients are copied verbatim from fdlibm.
#![allow(clippy::excessive_precision)]

const FRAC_2_PI: f64 = std::f64::consts::FRAC_2_PI;
// pi/2 split into three doubles; each FMA product below is exact.
const PIO2_HI: f64 = std::f64::consts::FRAC_PI_2;
const PIO2_MID: f64 = 6.123_233_995_736_766_036e-17;
const PIO2_LO: f64 = -1.497_384_904_859_169_724e-33;
// 1.5 * 2^52: adding it rounds to the nearest integer and leaves that
// integer in the low mantissa bits.
const ROUND_SHIFT: f64 = 6_755_399_441_055_744.0;

const S1: f64 = -1.666_666_666_666_663_243_48e-01;
const S2: f64 = 8.333_333_333_322_489_461_24e-03;
const S3: f64 = -1.984_126_982_985_794_931_34e-04;
const S4: f64 = 2.755_731_370_707_006_767_89e-06;
const S5: f64 = -2.505_076_025_340_686_341_95e-08;
const S6: f64 = 1.589_690_995_211_550_102_21e-10;

const C1: f64 = 4.166_666_666_666_660_190_37e-02;
const C2: f64 = -1.388_888_888_887_410_957_49e-03;
const C3: f64 = 2.480_158_728_947_672_941_78e-05;
const C4: f64 = -2.755_731_435_139_066_330_35e-07;
const C5: f64 = 2.087_572_321_298_174_827_90e-09;
const C6: f64 = -1.135_964_755_778_819_482_65e-11;

/// Returns `(sin(x), cos(x))`.
#[inline(always)]
pub fn sin_cos(x: f64) -> (f64, f64) {
    let shifted = x.mul_add(FRAC_2_PI, ROUND_SHIFT);
    let quadrant = shifted.to_bits();
    let n = shifted - ROUND_SHIFT;

    let r = (-n).mul_add(PIO2_HI, x);
    let r = (-n).mul_add(PIO2_MID, r);
    let r = (-n).mul_add(PIO2_LO, r);

    let z = r * r;
    let ps = z
        .mul_add(S6, S5)
        .mul_add(z, S4)
        .mul_add(z, S3)
        .mul_add(z, S2);
    let ps = ps.mul_add(z, S1);
    let s = (r * z).mul_add(ps, r);

    let pc = z
        .mul_add(C6, C5)
        .mul_add(z, C4)
        .mul_add(z, C3)
        .mul_add(z, C2);
    let pc = pc.mul_add(z, C1);
    let c = (z * z).mul_add(pc, (-0.5f64).mul_add(z, 1.0));

    // Quadrants 0..3 map to (s, c), (c, -s), (-s, -c), (-c, s).
    let swap = quadrant & 1 != 0;
    let (sin_r, cos_r) = if swap { (c, s) } else { (s, c) };
    let sin_sign = (quadrant & 2) << 62;
    let cos_sign = (quadrant.wrapping_add(1) & 2) << 62;
    (
        f64::from_bits(sin_r.to_bits() ^ sin_sign),
        f64::from_bits(cos_r.to_bits() ^ cos_sign),
    )
}

#[cfg(test)]
mod tests {
    use super::sin_cos;

    #[test]
    fn matches_libm() {
        let mut worst: f64 = 0.0;
        let mut x = -2.0e6;
        while x < 2.0e6 {
            let (s, c) = sin_cos(x);
            let (s_ref, c_ref) = x.sin_cos();
            worst = worst.max((s - s_ref).abs()).max((c - c_ref).abs());
            x += 0.987_654_321;
        }
        assert!(worst < 4.0e-16, "worst abs error {worst:e}");
    }
}
